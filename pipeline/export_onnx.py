import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from pipeline.train_ruri import RuriForRouting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_onnx")


def export_to_onnx(args):
    device = torch.device("cpu")  # Stage 2 は CPU 完結推論
    logger.info(f"Loading trained weights from {args.checkpoint_dir}")

    checkpoint_path = Path(args.checkpoint_dir) / "ruri_router_pytorch.pt"
    model = RuriForRouting(args.base_model_id, num_classes=3)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info(f"Loaded weights from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Exporting base model initialized head.")

    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)

    # ダミー入力の作成 (Batch=1, Seq=128)
    dummy_text = "トピック: 今日の天気を教えてください。"
    inputs = tokenizer(
        dummy_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    dummy_input_ids = inputs["input_ids"].to(device)
    dummy_attention_mask = inputs["attention_mask"].to(device)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ONNX model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=args.opset_version,
        do_constant_folding=True,
    )
    logger.info("ONNX export successful.")

    # INT8 量子化
    if args.quantize_int8:
        quantized_path = output_path.with_name(f"{output_path.stem}_int8.onnx")
        logger.info(f"Applying INT8 dynamic quantization -> {quantized_path}...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=str(output_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
            )
            logger.info(f"INT8 Quantization completed successfully. File: {quantized_path}")
        except ImportError:
            logger.warning("onnxruntime is not installed. Skipping INT8 quantization step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Ruri routing classifier to ONNX")
    parser.add_argument("--checkpoint_dir", type=str, default="models/ruri_classifier")
    parser.add_argument("--base_model_id", type=str, default="cl-nagoya/ruri-v3-30m")
    parser.add_argument("--output_path", type=str, default="models/router.onnx")
    parser.add_argument("--opset_version", type=int, default=17)
    parser.add_argument("--quantize_int8", action="store_true", default=True)
    args = parser.parse_args()
    export_to_onnx(args)
