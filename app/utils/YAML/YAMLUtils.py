import os
import yaml
import logging

class YAMLUtils:
    """Utility class for handling YAML files."""
    @staticmethod    
    def load_yaml(path: str) -> dict:
        """Load a YAML file into a Python dictionary."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logging.debug1(f"✅ Loaded YAML from: {path}")
        return data


    @staticmethod    
    def write_yaml(data: dict, path: str) -> None:
        """Dump a dictionary to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        logging.debug1(f"📝 YAML written to: {path}")


    @staticmethod    
    def update_yaml_key(path: str, keys: list, value) -> None:
        """Update or insert a nested key into a YAML file."""
        data = load_yaml(path)
        current = data
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
        write_yaml(data, path)
        logging.debug1(f"🔧 Updated key {'.'.join(keys)} in {path}")
