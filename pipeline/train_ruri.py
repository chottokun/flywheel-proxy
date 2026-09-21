import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_ruri")

LABEL_MAP = {"A": 0, "B": 1, "C": 2}
REVERSE_LABEL_MAP = {0: "A", 1: "B", 2: "C"}


class PromptDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                label_str = item.get("label", "").upper()
                if label_str not in LABEL_MAP:
                    continue

                # Ruri/ModernBERT 推奨のトピックプレフィックスを適用
                text = item.get("text", "")
                formatted_text = f"トピック: {text}"
                self.samples.append((formatted_text, LABEL_MAP[label_str]))

        logger.info(f"Loaded {len(self.samples)} valid samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class RuriForRouting(nn.Module):
    """
    cl-nagoya/ruri-v3-30m (ModernBERT-ja 37M) 基盤の Mean Pooling 分類ヘッド。
    """

    def __init__(self, base_model_id: str = "cl-nagoya/ruri-v3-30m", num_classes: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_id)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, Seq, Hidden]

        # Mean Pooling (attention_mask 考慮)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask  # [B, Hidden]

        logits = self.classifier(pooled)
        return logits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for training")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    dataset = PromptDataset(args.data_path, tokenizer, max_length=args.max_length)
    if len(dataset) == 0:
        logger.error("No samples found. Exiting.")
        return

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RuriForRouting(args.base_model_id, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "ruri_router_pytorch.pt")
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Training complete. Weights saved to {output_dir / 'ruri_router_pytorch.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ruri-v3-30m routing classifier")
    parser.add_argument("--data_path", type=str, default="data/labeled_dataset.jsonl")
    parser.add_argument("--base_model_id", type=str, default="cl-nagoya/ruri-v3-30m")
    parser.add_argument("--output_dir", type=str, default="models/ruri_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    train(args)
