
import argparse

from runners.expression_run import gene_expression_preprocessing
from utils.helpers import load_config, resolve_all_vars, flatten_config
from utils.model_utils import create_cross_validation_splits
from runners.training_run import train_model
from runners.test_run import test_model
from runners.visualisation_run import test_model_with_biomarkers
from utils.logging_utils import ExperimentLogger

def parser():

    parser = argparse.ArgumentParser(description="ProtoPathway")
    parser.add_argument("--config", type=str, default="configs/RA_config.yaml", help="Path to the configuration YAML file")

    args = parser.parse_args()

    # Load configuration
    config_raw = load_config(args.config)

    env = flatten_config(config_raw)

    # Resolve environment variables recursively
    resolved_env = resolve_all_vars(env, env)
    # Final config with nested resolution
    config = resolve_all_vars(config_raw, resolved_env)

    return config

def main(config, experiment_logger):

    logger = experiment_logger.logger
    logger.info(f"Using dataset: {config['dataset_name']}")

    if config['execution']['ge_preprocess']:
        logger.info("Gene Expression and Reactome Pathways preprocessing")
        gene_expression_preprocessing(config)
        logger.info("Gene Expression and Reactome Pathways preprocessing")

    if config['execution']['create_splits']:
        logger.info("Creating cross-validation splits")
        create_cross_validation_splits(config)
        logger.info("Cross-validation splits created")

    training_performed = False
    if config['execution']['cross_validation']:
        logger.info("Cross-validation of the model")
        train_model(config, experiment_logger=experiment_logger)
        logger.info("Cross-validation complete")
        training_performed = True

    if config['execution']['full_train']:
        logger.info("Full train of the model")
        train_model(config, is_full_train=True, experiment_logger=experiment_logger)
        logger.info("Training complete")
        training_performed = True

    if config['execution']['test']:
        logger.info("Testing the model")
        is_continuation = training_performed
        test_model(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
        logger.info("Testing complete")

    if config['execution']['visualise']:
        logger.info("Visualising model results")
        is_continuation = training_performed
        test_model_with_biomarkers(config, is_continuation=is_continuation, experiment_logger=experiment_logger)
        logger.info("Done visualising model results")


if __name__ == "__main__":

    # Parse command-line arguments
    config = parser()
    # Set up logging
    experiment_logger = ExperimentLogger(
        config,
        capture_console=True
    )

    main(config, experiment_logger)