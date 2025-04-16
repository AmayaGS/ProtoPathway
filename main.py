
import argparse

from runners.expression_run import gene_expression_preprocessing
from utils.helpers import load_config, resolve_all_vars

def parser():

    parser = argparse.ArgumentParser(description="ProtoPathway")
    parser.add_argument("--config", type=str, default="configs/RA_config.yaml", help="Path to the configuration YAML file")

    args = parser.parse_args()

    # Load configuration
    config_raw = load_config(args.config)

    env = {}
    for k, v in config_raw.items():
        if isinstance(v, str):
            env[k] = v

    # Resolve environment variables recursively
    resolved_env = resolve_all_vars(env, env)

    # Final config with nested resolution
    config = resolve_all_vars(config_raw, resolved_env)

    return config

def main(config):

    if config['execution']['ge_preprocess']:
        print("Gene Expression and Reactome Pathways preprocessing")

        # Run the preprocessing pipeline
        gene_expression_preprocessing(config)


if __name__ == "__main__":

    # Parse command-line arguments
    config = parser()
    main(config)