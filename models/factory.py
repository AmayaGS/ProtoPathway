"""
Model Factory for ProtoPathway.

Provides model-agnostic building interface:
    model = build_model(cfg, **kwargs)

Supports:
- ProtoPathway (multimodal, gene-only, wsi-only via config.model.branches)
- ABMIL, TransMIL, DSMIL (WSI baselines)
- SNN, MLP (Gene expression baselines)
- SurvPath, MMP, PIBD, MCAT, MOTCAT, PORPOISE
"""

import logging
from typing import Dict, Callable

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
# dsmil: none (WSI features only)
# snn: none (gene features only, uses bipartite graph)
# mlp: none (gene features flattened)
# =============================================================================


# -----------------------------------------------------------------------------
# ProtoPathway (main model)
# -----------------------------------------------------------------------------

@register_model("protopath")
def build_protopath(cfg, **kwargs):
    """
    Build ProtoPathway model from config.

    Args:
        cfg: OmegaConf config object
        wsi_centroids: Optional pre-computed centroids tensor

    Returns:
        ProtoPathway model instance
    """
    from models.protopath import ProtoPathway

    wsi_centroids = kwargs.get('wsi_centroids')

    # Determine number of classes
    if cfg.task == 'survival':
        num_classes = cfg.survival.num_bins
    else:
        num_classes = cfg.classification.num_classes

    model_cfg = cfg.model

    model = ProtoPathway(
        num_classes=num_classes,
        # Gene encoder
        gene_hidden_dim=model_cfg.gene_encoder.hidden_dim,
        gene_num_layers=model_cfg.gene_encoder.num_layers,
        gene_dropout=model_cfg.gene_encoder.dropout,
        gene_num_heads=model_cfg.gene_encoder.num_heads,
        gene_enabled=model_cfg.branches.gene,
        # WSI encoder
        wsi_input_dim=1536,
        wsi_hidden_dim=model_cfg.wsi_encoder.hidden_dim,
        wsi_num_prototypes=model_cfg.wsi_encoder.num_prototypes,
        wsi_tau=model_cfg.wsi_encoder.tau,
        wsi_centroids=wsi_centroids,
        wsi_enabled=model_cfg.branches.wsi,
        # Fusion
        fusion_type=model_cfg.fusion.type,
        fusion_num_heads=model_cfg.fusion.num_heads,
        fusion_dropout=model_cfg.fusion.dropout,
        fusion_output_dim=model_cfg.fusion.get('output_dim', 64),
    )

    return model

# -----------------------------------------------------------------------------
# WSI Baselines
# -----------------------------------------------------------------------------
@register_model("abmil")
def build_abmil(cfg, **kwargs):
    """Build Attention-Based MIL model."""
    from models.baselines.unimodal_wsi import ABMIL

    # Determine number of output classes
    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return ABMIL(input_dim=cfg.model.get('num_features', 1536), n_classes=n_classes)


@register_model("transmil")
def build_transmil(cfg, **kwargs):
    """Build TransMIL model."""
    from models.baselines.unimodal_wsi import TransMIL

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return TransMIL(
        num_features=cfg.model.get('num_features', 1536),
        n_classes=n_classes,
    )

@register_model("dsmil")
def build_dsmil(cfg, **kwargs):
    """Build DSMIL model."""
    from models.baselines.unimodal_wsi import DSMIL

    if cfg.task == "survival":
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return DSMIL(
        input_dim=cfg.model.get("num_features", 1536),
        n_classes=n_classes
    )

# -----------------------------------------------------------------------------
# Gene Expression Baselines
# -----------------------------------------------------------------------------
@register_model("snn")
def build_snn(cfg, **kwargs):
    """Build Survival Neural Network (SNN) for gene expression."""
    from models.baselines.unimodal_genes import SNN

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return SNN(
        num_genes=kwargs.get('num_genes', cfg.model.gene_encoder.get('num_genes')),
        hidden_dims=cfg.model.gene_encoder.get('hidden_dims', [256, 256]),
        n_classes=n_classes,
        dropout=cfg.model.wsi_encoder.dropout
    )


@register_model("mlp")
def build_mlp(cfg, **kwargs):
    """Build MLP baseline for gene expression."""
    from models.baselines.unimodal_genes import GeneExpressionMLP

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return GeneExpressionMLP(
        num_genes=kwargs.get('num_genes', cfg.model.gene_encoder.get('num_genes')),
        hidden_dims=cfg.model.gene_encoder.get('hidden_dims', [256, 256]),
        n_classes=n_classes,
        dropout=cfg.model.wsi_encoder.dropout # I'm using this because it's set to 0.25
    )

# -----------------------------------------------------------------------------
# Multimodal Baselines
# -----------------------------------------------------------------------------

@register_model("survpath")
def build_survpath(cfg, **kwargs):
    """
    Build SurvPath model from config.

    Args:
        cfg: Config object
        omic_sizes: List of gene counts per pathway
    """
    from models.baselines.survpath import SurvPath

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return SurvPath(
        num_features=cfg.model.get('wsi_input_dim', 1536),
        hidden_dim=cfg.model.wsi_encoder.get('hidden_dim', 256),
        num_heads=cfg.model.wsi_encoder.get('num_heads', 1),
        n_classes=n_classes,
        dropout=cfg.model.fusion.get('dropout', 0.25),
    )

@register_model("porpoise")
def build_porpoise(cfg, **kwargs):
    """Build PORPOISE model from config."""
    from models.baselines.porpoise import PORPOISE

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return PORPOISE(
        num_genes=kwargs.get('num_genes', cfg.model.gene_encoder.get('num_genes')),
        num_features=cfg.model.get('wsi_input_dim', 1536),
        fusion='bilinear',
        hidden_dim=cfg.model.wsi_encoder.get('hidden_dim', 256),
        n_classes=n_classes,
        dropout=cfg.model.fusion.get('dropout', 0.25),
        drop_input=0.10,
        gate_wsi=True,
        gate_omic=True,
        skip=True,
        use_mlp=False,
    )


@register_model("pibd")
def build_pibd(cfg, **kwargs):
    """Build PIBD model from config."""
    from models.baselines.pibd.pibd import PIBD

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return PIBD(num_features=cfg.model.get('wsi_input_dim', 1536),
                hidden_dim=256,
                n_classes=n_classes,
                bag_size=512,
                num_patches=4096,
                ratio_wsi=0.5,
                ratio_omics=0.5,
                sample_num=50,
                alpha=0.1,
                beta=0.01,
                seed=cfg.experiment.get('seed', 42))


@register_model("mmp")
def build_mmp_model(cfg, **kwargs):
    from models.baselines.mmp import MMP

    centroids = kwargs.get('wsi_centroids')
    if centroids is None:
        raise ValueError("MMP requires 'wsi_centroids' in model_kwargs. "
                        "Ensure needs_centroids=True and config has wsi settings.")

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return MMP(
        centroids=centroids,
        num_features=cfg.model.get('wsi_input_dim', 1536),
        hidden_dim=cfg.model.wsi_encoder.get('hidden_dim', 256),
        num_heads=cfg.model.wsi_encoder.get('num_heads', 1),
        n_classes=n_classes,
        dropout=cfg.model.fusion.get('dropout', 0.25),
        n_em_iters=1,
        tau=10
    )

@register_model("mcat")
def build_mcat_model(cfg, **kwargs):
    from models.baselines.mcat import MCAT

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return MCAT(
        num_features= cfg.model.get('wsi_input_dim', 1536),
        hidden_dim= cfg.model.wsi_encoder.get('hidden_dim', 256),
        num_heads= cfg.model.wsi_encoder.get('num_heads', 1),
        n_classes=n_classes,
        dropout= cfg.model.fusion.get('dropout', 0.25),
        fusion= 'concat',
        n_transformer_layers= 1
    )


@register_model("motcat")
def build_motcat_model(cfg, **kwargs):
    from models.baselines.motcat import MOTCAT

    if cfg.task == 'survival':
        n_classes = cfg.survival.num_bins
    else:
        n_classes = cfg.classification.num_classes

    return MOTCAT(
        num_features= cfg.model.get('wsi_input_dim', 1536),
        hidden_dim= cfg.model.wsi_encoder.get('hidden_dim', 256),
        num_heads= cfg.model.wsi_encoder.get('num_heads', 1),
        n_classes= n_classes,
        dropout= cfg.model.wsi_encoder.get('dropout', 0.25),
        fusion= 'concat',
        n_transformer_layers= 1,
        ot_impl= 'pot-uot-l2',
        ot_reg= 0.1,
        ot_tau= 0.5
    )
# -----------------------------------------------------------------------------
# Helper: Check model requirements
# -----------------------------------------------------------------------------
def get_model_requirements(model_name):
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
        'dsmil':{
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
        },
        'survpath': {
            'needs_graph': True,
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'multimodal'
        },
        'porpoise': {
            'needs_graph': False,
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'multimodal'
        },
        'pibd': {
            'needs_graph': True,  # Uses pathway structure
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'multimodal'
        },
        'mmp': {
            'needs_graph': True,
            'needs_wsi': True,
            'needs_centroids': True,
            'modality': 'multimodal'
        },
        'mcat': {
            'needs_graph': True,
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'multimodal'
        },
        'motcat': {
            'needs_graph': True,
            'needs_wsi': True,
            'needs_centroids': False,
            'modality': 'multimodal'
        }

    }
    return requirements.get(model_name.lower(), {})