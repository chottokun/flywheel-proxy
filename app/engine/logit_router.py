import asyncio
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.engine.base import BaseRouterEngine, ChatMessage

logger = logging.getLogger("flywheel.engine.logit_router")


class LogitRouterEngine(BaseRouterEngine):
    """
    chottokun/logit-router 準拠の Prefill Sliced LM-Head ルーター。
    メインイベントループをブロックしないよう asyncio.to_thread で推論を別スレッド実行する。
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        load_in_4bit: bool | None = None,
        load_in_8bit: bool | None = None,
    ):
        self.model_id = model_id or settings.ROUTER_MODEL_ID
        if device is None:
            self.device = settings.ROUTER_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        is_awq_model = "awq" in self.model_id.lower()
        is_gemma2 = "gemma" in self.model_id.lower()

        _load_4bit = settings.LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
        _load_8bit = settings.LOAD_IN_8BIT if load_in_8bit is None else load_in_8bit

        dtype = (
            "auto"
            if (_load_4bit or _load_8bit or is_awq_model)
            else (
                getattr(torch, settings.ROUTER_TORCH_DTYPE)
                if self.device == "cuda"
                else torch.float32
            )
        )

        quantization_config = None
        if _load_4bit or _load_8bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=_load_4bit,
                    load_in_8bit=_load_8bit,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed. Quantization disabled.")

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                attn_implementation="sdpa",
                **model_kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed with sdpa: {e}, falling back to default attn")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

        self.model.eval()
        self.backbone = self.model.model

        self.choice_letters = ["A", "B", "C"]
        self.routes = ["route_a", "route_b", "route_c"]

        # logit-router 実績パターン: 先頭空白付きトークン ID (" A", " B", " C")
        if is_gemma2:
            self.choice_token_ids = [
                self.tokenizer.encode(letter, add_special_tokens=False)[-1]
                for letter in self.choice_letters
            ]
        else:
            self.choice_token_ids = [
                self.tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]
                for letter in self.choice_letters
            ]

        choice_token_tensor = torch.tensor(self.choice_token_ids, device=self.device)
        self.is_sliced_head = False
        self.choice_head_weights = None
        if hasattr(self.model.lm_head, "weight") and self.model.lm_head.weight is not None:
            self.choice_head_weights = (
                self.model.lm_head.weight[choice_token_tensor].detach().clone()
            )
            self.is_sliced_head = True

        # 推論排他用セマフォ
        self._semaphore = asyncio.Semaphore(1)
        logger.info(
            f"LogitRouterEngine initialized: model={self.model_id}, device={self.device}, sliced={self.is_sliced_head}"
        )

    def _build_prompt(self, messages: list[ChatMessage], max_tokens: int = 2048) -> str:
        """システム指示と選択肢を保持し、トークン上限内で履歴を安全にパッキング"""
        system_prompts = [
            m.content for m in messages if m.role == "system" and isinstance(m.content, str)
        ]
        user_or_asst_msgs = [m for m in messages if m.role != "system"]

        system_instruction = (
            "You are a fast, high-precision request router. Classify the user's task difficulty "
            "strictly into one of three routes:\n"
            "A: Simple, routine, summary, translation, or casual conversation (Local LLM)\n"
            "B: Standard programming, data analysis, transformation, or structured reasoning (Fast Commercial LLM)\n"
            "C: Highly complex architecture, rigorous proof, subtle edge case, or uncertain request (Top Commercial LLM)\n"
            "Select the single best option letter."
        )
        if system_prompts:
            system_instruction += "\n\nAdditional System Instructions:\n" + "\n".join(system_prompts)

        history_lines = []
        for m in user_or_asst_msgs[-6:]:
            text = m.content if isinstance(m.content, str) else ""
            history_lines.append(f"[{m.role.upper()}]: {text.strip()}")

        context_text = "\n".join(history_lines)
        choices_text = "A. Route A (Local LLM)\nB. Route B (Fast Commercial LLM)\nC. Route C (Top Commercial LLM)"

        chat_messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": f"Context & Conversation:\n{context_text}\n\nChoices:\n{choices_text}\n\nSelect the single correct option letter.",
            },
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            if not prompt.endswith("Answer: "):
                prompt += "Answer: "
        except Exception:
            prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{context_text}\n\nChoices:\n{choices_text}\n\nSelect the single correct option letter.<|im_end|>\n<|im_start|>assistant\nAnswer: "

        return prompt

    @torch.inference_mode()
    def _sync_forward(self, prompt: str) -> dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        outputs = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state[:, -1, :]  # [1, hidden_dim]

        if self.is_sliced_head:
            logits = torch.matmul(last_hidden_state, self.choice_head_weights.t())
        else:
            full_logits = self.model.lm_head(last_hidden_state)
            choice_token_tensor = torch.tensor(self.choice_token_ids, device=self.device)
            logits = full_logits[:, choice_token_tensor]

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        prob_dict = {
            self.choice_letters[i]: float(probs[i]) for i in range(len(self.choice_letters))
        }

        sorted_probs = sorted(probs, reverse=True)
        # 正規化エントロピー H / ln(3)
        entropy = -sum(p * math.log(p + 1e-12) for p in sorted_probs) / math.log(3)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 1.0

        best_idx = probs.index(max(probs))
        raw_choice = self.choice_letters[best_idx]
        raw_route = self.routes[best_idx]

        # 不確実性エスカレーション
        escalated = False
        if entropy > settings.ENTROPY_THRESHOLD or margin < settings.MARGIN_THRESHOLD:
            final_route = "route_c"
            escalated = True
        else:
            final_route = raw_route

        return {
            "route": final_route,
            "raw_choice": raw_choice,
            "confidence": sorted_probs[0],
            "probs": prob_dict,
            "entropy": float(entropy),
            "margin": float(margin),
            "escalated": escalated,
        }

    async def determine_route(self, messages: list[ChatMessage]) -> tuple[str, dict[str, Any]]:
        prompt = self._build_prompt(messages)
        async with self._semaphore:
            # メインのイベントループをブロックしないよう別スレッドにオフロード
            metrics = await asyncio.to_thread(self._sync_forward, prompt)
        return metrics["route"], metrics
