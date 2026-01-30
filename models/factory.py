"""
Model Factory for ProtoPathway.

Provides model-agnostic building interface:
    model = build_model(cfg, **kwargs)

Supports:
- ProtoPathway (multimodal, gene-only, wsi-only via config.model.branches)
- ABMIL, TransMIL (WSI baselines)
- SNN, MLP (Gene expression baselines)
"""

import logging
from typing import Optional, Dict, Any, Callable

import torch.nn as nn

# Registry mapping model names to builder functions
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model builder."""
    def decorator(fn):
        MODEL_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def build_model(cfg, **kwargs) -> nn.Module:
    """
    Build model based on config.

    Args:
        cfg: OmegaConf config with model.name and related settings
        **kwargs: Additional arguments (e.g., wsi_centroids, graph_data)

    Returns:
        Initialized model
    """
    model_name = cfg.model.name.lower()

    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    builder = MODEL_REGISTRY[model_name]
    model = builder(cfg, **kwargs)

    logging.info(f"Built model: {model_name}")
    return model


def get_available_models():
    """Return list of registered model names."""
    return list(MODEL_REGISTRY.keys())


# =============================================================================
# Model Requirements (what each model needs from kwargs)
# =============================================================================
# protopath: wsi_centroids (optional), graph_data (implicit in dataset)
# abmil: none (WSI features only)
# transmil: none (WSI features only)
# snn: none (gene features only, uses bipartite graph)
# mlp: none (gene features flattened)
# =============================================================================


# -----------------------------------------------------------------------------
# ProtoPathway (main model)
# -----------------------------------------------------------------------------
@register_model("protopath")
def build_protopath(cfg, wsi_centroids=None, **kwargs):
    """Build ProtoPathway model."""
    from models.protopath import build_protopath as _build_protopath
    return _build_protopath(cfg, wsi_centroids=wsi_centroids)


# -----------------------------------------------------------------------------
# WSI Baselines
# -----------------------------------------------------------------------------
@register_model("abmil")
def build_abmil(cfg, **kwargs):
    """Build Attention-Based MIL model."""
    from models.baselines.wsi import ABMIL

    # Determine number of output classes
    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return ABMIL(
        input_dim=cfg.model.get('wsi_input_dim', 1536),  # UNI2-h default
        hidden_dim=cfg.model.wsi_encoder.hidden_dim,
        n_classes=n_classes,
        dropout=cfg.model.wsi_encoder.get('dropout', 0.1)
    )


@register_model("transmil")
def build_transmil(cfg, **kwargs):
    """Build TransMIL model."""
    from models.baselines.wsi import TransMIL

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return TransMIL(
        input_dim=cfg.model.get('wsi_input_dim', 1536),
        hidden_dim=cfg.model.wsi_encoder.hidden_dim,
        n_classes=n_classes,
        num_layers=cfg.model.wsi_encoder.get('num_layers', 2),
        num_heads=cfg.model.wsi_encoder.get('num_heads', 4),
        dropout=cfg.model.wsi_encoder.get('dropout', 0.1)
    )


# -----------------------------------------------------------------------------
# Gene Expression Baselines
# -----------------------------------------------------------------------------
@register_model("snn")
def build_snn(cfg, **kwargs):
    """Build Survival Neural Network (SNN) for gene expression."""
    from models.baselines.gene import SNN

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return SNN(
        num_genes=kwargs.get('num_genes', cfg.model.gene_encoder.get('num_genes')),
        hidden_dims=cfg.model.gene_encoder.get('hidden_dims', [256, 128]),
        n_classes=n_classes,
        dropout=cfg.model.gene_encoder.dropout
    )


@register_model("mlp")
def build_mlp(cfg, **kwargs):
    """Build MLP baseline for gene expression."""
    from models.baselines.gene import GeneExpressionMLP

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return GeneExpressionMLP(
        num_genes=kwargs.get('num_genes', cfg.model.gene_encoder.get('num_genes')),
        hidden_dims=cfg.model.gene_encoder.get('hidden_dims', [512, 256, 128]),
        n_classes=n_classes,
        dropout=cfg.model.gene_encoder.dropout
    )


# -----------------------------------------------------------------------------
# Helper: Check model requirements
# -----------------------------------------------------------------------------
def get_model_requirements(model_name: str) -> Dict[str, Any]:
    """
    Get requirements for a model type.

    Returns dict with:
        - needs_graph: Whether model uses bipartite graph
        - needs_wsi: Whether model uses WSI features
        - needs_centroids: Whether model needs pre-computed centroids
        - modality: 'gene', 'wsi', or 'multimodal'
    """
    requirements = {
        'protopath': {
            'needs_graph': True,
            'needs_wsi': True,  # Depends on cfg.model.branches
            'needs_centroids': True,  # Optional
            'modality': 'multimodal'
        },
        'abmil': {
            'needs_graph': False,
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'wsi'
        },
        'transmil': {
            'needs_graph': False,
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'wsi'
        },
        'snn': {
            'needs_graph': True,  # Uses pathway structure
            'needs_wsi': False,
            'needs_centroids': False,
            'modality': 'gene'
        },
        'mlp': {
            'needs_graph': False,  # Flattened genes
            'needs_wsi': False,
            'needs_centroids': False,
            'modality': 'gene'
        }
    }
    return requirements.get(model_name.lower(), {})