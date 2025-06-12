import os
import yaml
from pathlib import Path
from typing import Tuple, Optional, Any, Type
from dotenv import load_dotenv
from loguru import logger

from unsloth import FastLanguageModel
from datasets import load_dataset


from src.settings import (
    SetupModelConfig, 
    HyperparameterConfig
)


load_dotenv()
DEFAULT_CONFIG_PATH = Path("./configs/training.yaml")


def load_yaml_config(config_file: Optional[str] = None) -> dict:
    """Load and parse YAML configuration file."""
    config_path = Path(config_file) if config_file else DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}")


def add_specical_token(tokenizer, special_token: list):
    for i in special_token:
        tokenizer.add_tokens(i)
    return tokenizer


def setup_model(config_file: Optional[str] = None) -> Tuple[Any, Any]:
    """Setup and load the language model with tokenizer."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required but not set")
    
    yaml_config = load_yaml_config(config_file)
    
    if 'setup_model' not in yaml_config:
        raise KeyError("'setup_model' section not found in configuration file")
    
    training_config = SetupModelConfig(**yaml_config['setup_model'])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=training_config.model_id,
        max_seq_length=training_config.max_seq_length,
        dtype=training_config.dtype,
        full_finetuning=training_config.full_finetuning,
        token=hf_token,
    )

    tokenizer = add_specical_token(tokenizer, ['<metadata>', "</metadata>"])

    return model, tokenizer


def load_hyperparameters(
    config_file: Optional[str] = None,
    config_class: Type[HyperparameterConfig] = HyperparameterConfig
) -> HyperparameterConfig:
    """
    Load hyperparameters from a YAML config using a specified config class.
    Default is HyperparameterConfig, but can pass subclass (e.g., LlamaHyperparameterConfig).
    """
    if config_file is None:
        print(f"No config file provided. Using default {config_class.__name__} settings.")
        return config_class()

    yaml_config = load_yaml_config(config_file)

    if not isinstance(yaml_config, dict):
        raise ValueError("Configuration file content must be a dictionary at the top level.")

    if 'hyperparameters' not in yaml_config:
        raise KeyError("'hyperparameters' section not found in configuration file.")

    return config_class(**yaml_config['hyperparameters'])


def setup(config_file: Optional[str] = None) -> Tuple[Any, Any]:
    """Legacy setup function. Use setup_model() instead."""
    import warnings
    warnings.warn(
        "setup() is deprecated, use setup_model() instead", 
        DeprecationWarning, 
        stacklevel=2
    )
    return setup_model(config_file)


def log_hyperparameters(args, message="Hyperparameters:"):
    logger.info(message)
    for key, value in args.__dict__.items():
        logger.info("  {}: {}", key, value)


def format_metadata(metadata: dict) -> str:
    s = "<metadata>\n"
    for k, v in metadata.items():
        s += f"{k}: {v}\n"
    s += "</metadata>"
    return s

