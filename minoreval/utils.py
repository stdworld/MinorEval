import yaml
from pathlib import Path
from typing import Dict, Any

# --- Project Root Definition ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args: file_path: The path to the YAML file, relative to the project root.

    Returns: A dictionary containing the YAML content. Raises an exception on failure.
    """

    full_path = PROJECT_ROOT / file_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file at {full_path}: {e}")
    except Exception as e:
        raise IOError(f"An error occurred while loading config file {full_path}: {e}")

def truncate_text(text: str, max_length: int = 40) -> str:
    """
    Truncates text to a specified maximum length.
    """
    if not isinstance(text, str):
        return str(text)
    # Remove newlines to keep logs clean and single-lined.
    text = text.replace('\n', ' ').replace('\r', '')
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text