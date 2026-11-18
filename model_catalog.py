"""Resolve model definitions for workers outside the MaiBot process."""
from pathlib import Path

import os
import tomllib


def read_model_config(root: Path) -> dict:
    with (root / 'config/model_config.toml').open('rb') as stream:
        config = tomllib.load(stream)
    catalog_env = os.environ.get('MAIBOT_SHARED_MODEL_CATALOG')
    catalog_path = Path(catalog_env) if catalog_env else root / 'host/model_catalog.toml'
    if catalog_env or catalog_path.is_file():
        with catalog_path.open('rb') as stream:
            catalog = tomllib.load(stream)
        for key in ('models', 'api_providers'):
            config[key] = catalog[key]
    return config
