import os
import re
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def flatten_config(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep=sep))
        elif isinstance(v, str):
            items[new_key] = v
    return items


def resolve_all_vars(obj, env):
    """
    Recursively resolve all ${var} expressions using env,
    even if nested references exist.
    """
    pattern = re.compile(r"\$\{([^}^{]+)\}")

    def substitute(value):
        while isinstance(value, str) and pattern.search(value):
            value = pattern.sub(lambda m: env.get(m.group(1), m.group(0)), value)
        return value

    if isinstance(obj, dict):
        return {k: resolve_all_vars(v, env) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_all_vars(i, env) for i in obj]
    elif isinstance(obj, str):
        return substitute(obj)
    else:
        return obj


def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path