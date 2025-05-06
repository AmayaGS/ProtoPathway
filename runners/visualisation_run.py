import pickle
import pandas as pd
import os

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from utils.helpers import ensure_directory
from utils.model_utils import initialise_model
from utils.dataset_utils import build_incidence_matrix, HypergraphDataset
from utils.biomarker_analysis import BiomarkerAnalysis, integrate_with_evaluate_model
from models.ProtoPathway import PathwayEmbeddingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model_with_biomarkers(config, is_continuation=False, experiment_logger=None):
    logger = experiment_logger.logger

    # Set up directories
    if is_continuation:
        results_dir = experiment_logger.log_dir
        model_dir = experiment_logger.checkpoint_dir
    else:
        results_dir = config['testing']['experiment_path']
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Experiment directory not found: {results_dir}")
        model_dir = os.path.join(results_dir, "checkpoints")

    # Create test results and biomarker analysis directories
    test_results_dir = os.path.join(results_dir, 'test_results')
    biomarker_dir = os.path.join(test_results_dir, 'biomarker_analysis')
    ensure_directory(test_results_dir)
    ensure_directory(biomarker_dir)

    logger.info(f"Starting model testing with biomarker analysis")
    logger.info(f"Results will be saved to {test_results_dir}")

    # Load data
    gene_expression_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)
    labels_df = pd.read_csv(
        os.path.join(config['output']['data']['dir'], f"patient_labels_{config['dataset_name']}.csv"))

    # Load splits
    with open(os.path.join(config['output']['data']['dir'],
                           f"data_splits_{config['dataset_name']}.pkl"), "rb") as f:
        split_dict = pickle.load(f)

    # Build incidence matrix and prepare graph structure
    pathway_data = build_incidence_matrix(
        config['output']['data']['final_pathways'],
        gene_expression_df
    )

    # Load the test set
    test_data = gene_expression_df.loc[split_dict["Test"]]
    test_dataset = HypergraphDataset(config, test_data, labels_df, pathway_data)
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=1,  # Use batch size of 1 for per-patient analysis
        num_workers=config['training']['num_workers'],
        shuffle=False
    )

    # Load model
    checkpoint_name = "best_fold_0.pt"
    model_path = os.path.join(model_dir, checkpoint_name)

    # Initialize model with name mappings
    model = PathwayEmbeddingModel(
        in_channels=1,
        hidden_channels=100,
        out_channels=config['n_classes'],
        num_layers=3,
        dropout=0.2,
        gene_names=pathway_data['gene_names'],
        pathway_names=pathway_data['pathway_names']
    )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Run evaluation with biomarker analysis
    logger.info("Evaluating model and analyzing biomarkers...")
    results = integrate_with_evaluate_model(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=biomarker_dir
    )

    # Log results
    accuracy = results['acc'] / 100.0
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Biomarker analysis report generated at: {results['biomarker_analysis']['report_paths']['report']}")

    # Extract significant biomarkers
    analyzer = results['biomarker_analysis']['analyzer']
    sig_pathways = analyzer.biomarker_results['pathway_biomarkers'][
        analyzer.biomarker_results['pathway_biomarkers']['significant']
    ]
    sig_genes = analyzer.biomarker_results['gene_biomarkers'][
        analyzer.biomarker_results['gene_biomarkers']['significant']
    ]

    logger.info(f"Identified {len(sig_pathways)} significant pathway biomarkers")
    logger.info(f"Identified {len(sig_genes)} significant gene biomarkers")

    # Save prediction results
    prediction_df = pd.DataFrame({
        'patient_id': results['patient_ids'],
        'true_label': results['all_targets'],
        'predicted_label': results['all_preds']
    })

    # Add class names if available
    if 'label_dict' in config:
        prediction_df['true_class'] = prediction_df['true_label'].map(
            lambda x: config['label_dict'].get(str(x), f"Class_{x}")
        )
        prediction_df['predicted_class'] = prediction_df['predicted_label'].map(
            lambda x: config['label_dict'].get(str(x), f"Class_{x}")
        )

    # Save predictions
    prediction_df.to_csv(os.path.join(test_results_dir, 'patient_predictions.csv'), index=False)

    # Add top biomarkers to structured results
    structured_results = {
        'metrics': {
            'accuracy': float(accuracy),
            'loss': float(results['loss']),
        },
        'top_pathway_biomarkers': sig_pathways.head(10).to_dict('records') if len(sig_pathways) > 0 else [],
        'top_gene_biomarkers': sig_genes.head(10).to_dict('records') if len(sig_genes) > 0 else [],
        'report_paths': results['biomarker_analysis']['report_paths']
    }

    # Save structured results
    import json
    with open(os.path.join(test_results_dir, 'test_results.json'), 'w') as f:
        json.dump(structured_results, f, indent=2)

    return structured_results