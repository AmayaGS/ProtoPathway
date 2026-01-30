"""
I/O utilities for file handling.
"""

import os
import logging


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.debug(f"Created directory: {path}")


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_schema(model, cfg):
    """
    Generate a detailed model schema with layer info and dimensions.

    Returns a dict with:
    - model_name: Name of the model
    - total_params: Total trainable parameters
    - branches: Which branches are enabled (gene/wsi)
    - layers: List of layer info dicts
    """
    schema = {
        'model_name': cfg.model.name,
        'branches': {
            'gene': cfg.model.branches.gene,
            'wsi': cfg.model.branches.wsi
        },
        'task': cfg.task,
        'total_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_params_all': sum(p.numel() for p in model.parameters()),
        'layers': []
    }

    # Iterate through named modules
    for name, module in model.named_modules():
        if name == '':  # Skip root module
            continue

        layer_info = {
            'name': name,
            'type': module.__class__.__name__,
            'params': sum(p.numel() for p in module.parameters(recurse=False)),
            'trainable': sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad),
        }

        # Extract dimensions for common layer types
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            layer_info['shape'] = f"({module.in_features}, {module.out_features})"
        elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            layer_info['shape'] = f"({module.in_channels}, {module.out_channels})"
            if hasattr(module, 'kernel_size'):
                layer_info['kernel_size'] = str(module.kernel_size)
        elif hasattr(module, 'num_features'):
            layer_info['shape'] = f"({module.num_features},)"
        elif hasattr(module, 'normalized_shape'):
            layer_info['shape'] = str(module.normalized_shape)
        elif hasattr(module, 'embedding_dim'):
            layer_info['shape'] = f"(embed_dim={module.embedding_dim})"
            if hasattr(module, 'num_embeddings'):
                layer_info['shape'] = f"({module.num_embeddings}, {module.embedding_dim})"

        # Only include layers with parameters or meaningful structure
        if layer_info['params'] > 0 or module.__class__.__name__ in [
            'GATv2Conv', 'Sequential', 'ModuleList', 'Dropout', 'ReLU', 'GELU', 'Softmax'
        ]:
            schema['layers'].append(layer_info)

    return schema


def format_model_schema(schema):
    """Format model schema as readable string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"MODEL ARCHITECTURE: {schema['model_name']}")
    lines.append("=" * 80)
    lines.append(f"Task: {schema['task']}")
    if 'branches' in schema:
        lines.append(f"Branches: Gene={schema['branches']['gene']}, WSI={schema['branches']['wsi']}")
    lines.append(f"Total Trainable Parameters: {schema['total_params']:,}")
    lines.append(f"Total Parameters (all): {schema['total_params_all']:,}")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'Layer Name':<45} {'Type':<20} {'Params':>10} {'Shape':<15}")
    lines.append("-" * 80)

    for layer in schema['layers']:
        shape = layer.get('shape', '')
        lines.append(
            f"{layer['name']:<45} {layer['type']:<20} {layer['params']:>10,} {shape:<15}"
        )

    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


def setup_logging_to_file(output_dir):
    """Add file handler to save all logs to a file."""
    log_path = os.path.join(output_dir, 'training_log.txt')

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # Use same format as console
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add to root logger
    logging.getLogger().addHandler(file_handler)

    return log_path, file_handler
