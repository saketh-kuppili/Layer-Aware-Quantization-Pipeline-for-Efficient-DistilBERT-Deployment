"""
Layer-wise sensitivity analysis.

For each quantizable component, quantizes ONLY that component to INT8
while keeping everything else at FP32, then measures accuracy
to determine which components are most sensitive to quantization.
"""

import torch
import torch.nn as nn
from copy import deepcopy


def quantize_single_layer(model, target_layer_name):
    """
    Quantize a single module (and all its Linear children) to INT8,
    leaving all other modules at FP32.

    Parameters
    ----------
    model : nn.Module
        The FP32 model (will be deep-copied, not modified).
    target_layer_name : str
        Dot-separated path to the module.

    Returns
    -------
    nn.Module
        Model with only the target module's Linear layers quantized.
    """
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

    # Quantize all Linear layers within this module
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

    Parameters
    ----------
    model : nn.Module
        The model to inspect.
    max_layers : int, optional
        Limit the number of components returned.
    granularity : str
        'block' — full transformer blocks + classifier head (recommended).
        'layer' — individual nn.Linear layers.

    Returns
    -------
    list[str]
        Component names suitable for sensitivity analysis.
    """
    if granularity == "block":
        layers = []

        # DistilBERT transformer blocks
        if hasattr(model, "distilbert"):
            transformer = model.distilbert.transformer
            for i in range(len(transformer.layer)):
                block = f"distilbert.transformer.layer.{i}"
                layers.append(block)

                # Also add sub-components for finer analysis
                layers.append(f"{block}.attention")
                layers.append(f"{block}.ffn")

            # Classifier head
            if hasattr(model, "pre_classifier"):
                layers.append("pre_classifier")
            if hasattr(model, "classifier"):
                layers.append("classifier")
        else:
            # Fallback: individual Linear layers
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

    For each component: quantize ONLY that component, measure accuracy,
    compare with FP32 baseline.

    Parameters
    ----------
    model : nn.Module
        The FP32 baseline model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding text.
    texts : list[str]
        Evaluation sentences.
    labels : list[int]
        Ground truth labels.
    layer_names : list[str]
        Components to analyze.
    benchmark_fn : callable
        Function with signature (model, tokenizer, texts, labels) -> dict.

    Returns
    -------
    tuple
        (results_dict, baseline_accuracy)
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