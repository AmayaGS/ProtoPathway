
import argparse

from runners.expression_run import gene_expression_preprocessing
from utils.helpers import load_config, resolve_all_vars, flatten_config
from utils.model_utils import create_cross_validation_splits
from runners.cross_validation_run import cross_validation

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

def main(config):

    if config['execution']['ge_preprocess']:
        print("Gene Expression and Reactome Pathways preprocessing")
        gene_expression_preprocessing(config)
        print("Gene Expression and Reactome Pathways preprocessing")

    if config['execution']['create_splits']:
        print("Creating cross-validation splits")
        create_cross_validation_splits(config)
        print("Cross-validation splits created")

    if config['execution']['cross_validation']:
        print("Cross-validation of the model")
        cross_validation(config)
        print("Cross-validation complete")

    if config['execution']['full_train']:
        print("Cross-validation of the model")
        cross_validation(config)
        print("Cross-validation complete")



if __name__ == "__main__":

    # Parse command-line arguments
    config = parser()
    main(config)