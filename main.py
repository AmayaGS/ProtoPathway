import argparse
import os

from utils.helpers import load_config, resolve_all_vars, flatten_config
from utils.model_utils import create_cross_validation_splits
from utils.logging_utils import ExperimentLogger

# Import GE-specific runners
from runners.gene_preprocessing import gene_expression_preprocessing
from runners.gene_training_run import train_gene_expression_model
from runners.gene_test_run import test_gene_expression_model

# Import WSI-specific runners
from runners.wsi_preprocessing import wsi_preprocessing
from runners.wsi_training_run import train_wsi_model
# from runners.wsi_test_run import test_wsi_model

# # Import multimodal runners
# from runners.multimodal_run import train_multimodal_model
# from runners.multimodal_test_run import test_multimodal_model


def parser():
    parser = argparse.ArgumentParser(description="ProtoPathway")
    parser.add_argument("--config", type=str, default="configs/R4RA_config.yaml",
                        help="Path to the configuration YAML file")
    parser.add_argument("--mode", type=str,
                        choices=["gene_expression", "wsi", "multimodal"],
                        help="Override the execution mode from config")

    args = parser.parse_args()

    # Load configuration
    config_raw = load_config(args.config)

    env = flatten_config(config_raw)

    # Resolve environment variables recursively
    resolved_env = resolve_all_vars(env, env)
    # Final config with nested resolution
    config = resolve_all_vars(config_raw, resolved_env)

    # Override mode if specified in command line
    if args.mode:
        config['execution']['mode'] = args.mode

    return config


def main(config, experiment_logger):
    logger = experiment_logger.logger
    logger.info(f"Using dataset: {config['dataset_name']}")
    logger.info(f"Execution mode: {config['execution']['mode']}")

    # Ensure output directories exist
    os.makedirs(config['output']['data']['dir'], exist_ok=True)
    os.makedirs(config['output']['figures']['dir'], exist_ok=True)

    # Common preprocessing steps regardless of mode
    if config['execution']['create_splits']:
        logger.info("Creating cross-validation splits")
        create_cross_validation_splits(config)
        logger.info("Cross-validation splits created")

    # Execute mode-specific preprocessing
    execution_mode = config['execution']['mode']

    if execution_mode == "gene_expression" or execution_mode == "multimodal":
        if config['execution']['ge_preprocess'] and config['gene_expression']['enabled']:
            logger.info("Gene Expression and Reactome Pathways preprocessing")
            gene_expression_preprocessing(config)
            logger.info("Gene Expression preprocessing complete")

    if execution_mode == "wsi" or execution_mode == "multimodal":
        if config['execution']['wsi_preprocess'] and config['wsi']['enabled']:
            logger.info("WSI preprocessing")
            wsi_preprocessing(config)
            logger.info("WSI preprocessing complete")

    # Execute mode-specific ge_training and testing
    training_performed = False

    # Gene Expression mode
    if execution_mode == "gene_expression":
        if config['execution']['cross_validation']:
            logger.info("Cross-validation of the gene expression model")
            train_gene_expression_model(config, experiment_logger=experiment_logger)
            logger.info("Cross-validation complete")
            training_performed = True

        if config['execution']['full_train']:
            logger.info("Full ge_training of the gene expression model")
            train_gene_expression_model(config, is_full_train=True, experiment_logger=experiment_logger)
            logger.info("Training complete")
            training_performed = True

        if config['execution']['test']:
            logger.info("Testing the gene expression model")
            is_continuation = training_performed
            test_gene_expression_model(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
            logger.info("Testing complete")

        if config['execution']['visualise']:
            logger.info("Visualising the gene expression and pathway biomarkers")
            is_continuation = training_performed
            test_gene_expression_model(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
            logger.info("Visualsation complete")

        # if config['execution']['visualise']:
        #     logger.info("Visualizing gene expression model results")
        #     is_continuation = training_performed
        #     test_ge_model_with_biomarkers(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
        #     logger.info("Visualization complete")

    # # WSI mode
    elif execution_mode == "wsi":
        if config['execution']['cross_validation']:
            logger.info("Cross-validation of the WSI model")
            train_wsi_model(config, experiment_logger=experiment_logger)
            logger.info("Cross-validation complete")
            training_performed = True

        if config['execution']['full_train']:
            logger.info("Full ge_training of the WSI model")
            train_wsi_model(config, is_full_train=True, experiment_logger=experiment_logger)
            logger.info("Training complete")
            training_performed = True
    #
    #     if config['execution']['test']:
    #         logger.info("Testing the WSI model")
    #         is_continuation = training_performed
    #         test_wsi_model(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
    #         logger.info("Testing complete")

    # # Multimodal mode
    # elif execution_mode == "multimodal":
    #     if config['execution']['cross_validation']:
    #         logger.info("Cross-validation of the multimodal model")
    #         train_multimodal_model(config, experiment_logger=experiment_logger)
    #         logger.info("Cross-validation complete")
    #         training_performed = True
    #
    #     if config['execution']['full_train']:
    #         logger.info("Full ge_training of the multimodal model")
    #         train_multimodal_model(config, is_full_train=True, experiment_logger=experiment_logger)
    #         logger.info("Training complete")
    #         training_performed = True
    #
    #     if config['execution']['test']:
    #         logger.info("Testing the multimodal model")
    #         is_continuation = training_performed
    #         test_multimodal_model(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
    #         logger.info("Testing complete")
    #
    # else:
    #     logger.error(f"Unknown execution mode: {execution_mode}")
    #     return


if __name__ == "__main__":
    # Parse command-line arguments
    config = parser()

    # Set up logging
    experiment_logger = ExperimentLogger(
        config,
        capture_console=True
    )

    main(config, experiment_logger)