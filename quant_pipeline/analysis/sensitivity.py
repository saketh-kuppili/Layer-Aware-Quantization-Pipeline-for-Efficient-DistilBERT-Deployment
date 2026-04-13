"""
Layer-wise sensitivity analysis.

For each quantizable component, quantizes ONLY that component to INT8
while keeping everything else at FP32, then measures accuracy
to determine which components are most sensitive to quantization.
"""

import torch
import torch.nn as nn
import platform
from copy import deepcopy


def _set_quantization_backend():
    """Set the appropriate quantization backend for the current platform."""
    system = platform.system()
    processor = platform.processor()
    is_mac = system == "Darwin"
    is_arm = "arm" in processor.lower() or "apple" in processor.lower()

    if is_mac and is_arm:
        torch.backends.quantized.engine = "qnnpack"
    else:
        torch.backends.quantized.engine = "fbgemm"


def quantize_single_layer(model, target_layer_name):
    """
    Quantize a single module (and all its Linear children) to INT8,
    leaving all other modules at FP32.
    """
    _set_quantization_backend()

    modified = deepcopy(model)
    modified.eval()

    parts = target_layer_name.split(".")
    parent = modified
    for part in parts[:-1]:
        if hasattr(parent, part):
            parent = getattr(parent, part)
        elif part.isdigit():
            parent = parent[int(part)]
        else:
            raise ValueError(f"Cannot navigate to '{part}' in '{target_layer_name}'")

    target = getattr(parent, parts[-1])

    quantized = torch.quantization.quantize_dynamic(
        target,
        {nn.Linear},
        dtype=torch.qint8,
    )
    setattr(parent, parts[-1], quantized)

    return modified


def get_quantizable_layers(model, max_layers=None, granularity="block"):
    """
    Get names of quantizable components in the model.
    """
    if granularity == "block":
        layers = []

        if hasattr(model, "distilbert"):
            transformer = model.distilbert.transformer
            for i in range(len(transformer.layer)):
                block = f"distilbert.transformer.layer.{i}"
                layers.append(block)
                layers.append(f"{block}.attention")
                layers.append(f"{block}.ffn")

            if hasattr(model, "pre_classifier"):
                layers.append("pre_classifier")
            if hasattr(model, "classifier"):
                layers.append("classifier")
        else:
            layers = [
                name for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
            ]

    else:
        layers = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

    if max_layers:
        layers = layers[:max_layers]
    return layers


def analyze_sensitivity(model, tokenizer, texts, labels, layer_names, benchmark_fn):
    """
    Run layer-wise sensitivity analysis.
    """
    print("Computing FP32 baseline...")
    baseline = benchmark_fn(model, tokenizer, texts, labels)
    baseline_acc = baseline["accuracy"]
    print(f"Baseline accuracy: {baseline_acc:.4f}\n")

    results = {}

    for i, layer_name in enumerate(layer_names):
        print(f"[{i + 1}/{len(layer_names)}] Quantizing: {layer_name}")

        try:
            modified_model = quantize_single_layer(model, layer_name)
            metrics = benchmark_fn(modified_model, tokenizer, texts, labels)
            acc = metrics["accuracy"]
            delta = acc - baseline_acc

            results[layer_name] = {
                "accuracy": acc,
                "delta": delta,
            }
            print(f"  accuracy={acc:.4f} (delta={delta:+.4f})\n")

        except Exception as e:
            print(f"  Skipping: {e}\n")
            results[layer_name] = {"accuracy": baseline_acc, "delta": 0.0}

    return results, baseline_acc