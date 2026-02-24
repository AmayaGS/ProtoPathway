"""
ProtoPathway

Usage:
    # Preprocessing
    python main.py preprocess pathways --config configs/preprocessing/preprocess_pathways.yaml
    python main.py preprocess genes --config configs/preprocessing/preprocess_genes.yaml
    python main.py preprocess wsi --config configs/preprocessing/preprocess_wsi.yaml
    python main.py preprocess splits --config configs/preprocessing/create_splits.yaml

    # Training
    python main.py train --config configs/experiments/experiment.yaml
    python main.py train --config configs/experiments/experiment.yaml dataset=BLCA training.lr=1e-5

    # WSI baselines
    python main.py train --config configs/experiments/experiment.yaml model.name=abmil
    python main.py train --config configs/experiments/experiment.yaml model.name=transmil

    # Gene baselines
    python main.py train --config configs/experiments/experiment.yaml model.name=snn
    python main.py train --config configs/experiments/experiment.yaml model.name=mlp

    # Evaluation
    python main.py evaluate --checkpoint-dir output/BLCA/exp_001

    # Visualization
    python main.py visualize --eval_dir output/BLCA/exp_001/evaluate
"""

import argparse
import os
import logging
from omegaconf import OmegaConf


def setup_logging(level=logging.INFO):
    """Configure logging for the entire application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ProtoPathway")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # -------------------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------------------
    prep_p = subparsers.add_parser('preprocess', help='Run preprocessing steps')
    prep_sub = prep_p.add_subparsers(dest='step', required=True)

    # preprocess pathways
    prep_reactome = prep_sub.add_parser('pathways', help='Process Reactome pathways (run once)')
    prep_reactome.add_argument('--config', default='configs/preprocessing/preprocess_pathways.yaml')

    # preprocess genes
    prep_genes = prep_sub.add_parser('genes', help='Preprocess gene expression data')
    prep_genes.add_argument('--config', default='configs/preprocessing/preprocess_genes.yaml')

    # preprocess wsi
    prep_wsi = prep_sub.add_parser('wsi', help='Preprocess WSI features')
    prep_wsi.add_argument('--config', default='configs/preprocessing/preprocess_wsi.yaml')

    # preprocess splits
    prep_splits = prep_sub.add_parser('splits', help='Create/load data splits')
    prep_splits.add_argument('--config', default='configs/preprocessing/create_splits.yaml')

    # preprocess all
    prep_all = prep_sub.add_parser('all', help='Run all preprocessing steps')
    prep_all.add_argument('--reactome-config', default='configs/preprocessing/preprocess_pathways.yaml')
    prep_all.add_argument('--gene-config', default='configs/preprocessing/preprocess_genes.yaml')
    prep_all.add_argument('--wsi-config', default='configs/preprocessing/preprocess_wsi.yaml')
    prep_all.add_argument('--splits-config', default='configs/preprocessing/create_splits.yaml')

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    train_p = subparsers.add_parser('train', help='Train a model')
    train_p.add_argument('--config', default='configs/experiments/experiment.yaml')
    train_p.add_argument('--device', default='cuda')
    train_p.add_argument('overrides', nargs='*', help='Config overrides (e.g., model_name=abmil)')

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--checkpoint-dir', type=str, required=True,
                             help='Experiment directory containing fold checkpoints')
    eval_parser.add_argument('--config', type=str, default=None,
                             help='Config file (auto-loads from checkpoint-dir if not provided)')

    # -------------------------------------------------------------------------
    # Visualize
    # -------------------------------------------------------------------------
    vis_p = subparsers.add_parser('visualize', help='Generate visualizations')
    vis_p.add_argument('--eval_dir', required=True, help='Path to experiment results directory')
    vis_p.add_argument('--output', default=None, help='Output directory (defaults to results/figures)')

    return parser.parse_args()


def load_config(config_path, overrides=None):
    """Load config with OmegaConf and apply CLI overrides."""
    cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    return cfg


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    if args.command == 'preprocess':

        if args.step == 'pathways':
            from preprocessing.preprocess_reactome import run
            cfg = load_config(args.config)
            run(cfg)

        elif args.step == 'genes':
            from preprocessing.preprocess_genes import run
            cfg = load_config(args.config)
            run(cfg)

        elif args.step == 'wsi':
            from preprocessing.preprocess_wsi import run
            cfg = load_config(args.config)
            run(cfg)

        elif args.step == 'splits':
            from preprocessing.create_splits import run
            cfg = load_config(args.config)
            run(cfg)

        elif args.step == 'all':
            print("=" * 60)
            print("Step 1/4: Processing Reactome pathways")
            print("=" * 60)
            from preprocessing.preprocess_reactome import run as run_reactome
            cfg_reactome = load_config(args.reactome_config)
            run_reactome(cfg_reactome)

            print("\n" + "=" * 60)
            print("Step 2/4: Preprocessing gene expression")
            print("=" * 60)
            from preprocessing.preprocess_genes import run as run_genes
            cfg_genes = load_config(args.gene_config)
            run_genes(cfg_genes)

            print("\n" + "=" * 60)
            print("Step 3/4: Preprocessing WSI features")
            print("=" * 60)
            from preprocessing.preprocess_wsi import run as run_wsi
            cfg_wsi = load_config(args.wsi_config)
            run_wsi(cfg_wsi)

            print("\n" + "=" * 60)
            print("Step 4/4: Creating data splits")
            print("=" * 60)
            from preprocessing.create_splits import run as run_splits
            cfg_splits = load_config(args.splits_config)
            run_splits(cfg_splits)

            print("\n" + "=" * 60)
            print("Preprocessing complete!")
            print("=" * 60)

    elif args.command == 'train':
        from experiments.train import run
        cfg = load_config(args.config, args.overrides)
        cfg.device = args.device
        run(cfg)

    elif args.command == 'evaluate':
        from experiments.evaluate import run

        if args.config is None:
            config_path = os.path.join(args.checkpoint_dir, 'config.yaml')
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"No config found at {config_path}. Please provide --config."
                )
            cfg = load_config(config_path)
        else:
            cfg = load_config(args.config)

        cfg.checkpoint_dir = args.checkpoint_dir
        run(cfg)

    elif args.command == 'visualize':
        from experiments.visualize_v3 import run_simplified_visualization

        eval_dir = args.eval_dir
        output_dir = args.output

        # Try loading entity names from experiment config
        # (fallback: they'll be read from attention pickle metadata)
        entity_names = None
        parent_dir = os.path.dirname(eval_dir.rstrip('/'))
        config_path = os.path.join(parent_dir, 'config.yaml')

        if os.path.exists(config_path):
            from utils.dataset import load_dataset_components
            cfg = load_config(config_path)
            data_components = load_dataset_components(cfg)
            graph_data = data_components['graph_data']
            entity_names = {
                'gene_names': graph_data.get('gene_names', []),
                'pathway_names': graph_data.get('pathway_names', []),
            }
        else:
            logging.info(
                "No config.yaml found — entity names will be loaded from "
                "attention pickle metadata"
            )

        run_simplified_visualization(
            eval_dir=eval_dir,
            output_dir=output_dir,
            entity_names=entity_names,
        )


if __name__ == '__main__':
    main()