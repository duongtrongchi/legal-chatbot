import os
import pytest
from src import utils
from src.settings import HyperparameterConfig

def test_load_yaml_config():
    config = utils.load_yaml_config("configs/training.yaml")
    assert "setup_model" in config
    assert "hyperparameters" in config

def test_load_hyperparameters():
    hparams = utils.load_hyperparameters("configs/training.yaml")
    assert isinstance(hparams, HyperparameterConfig)
    assert hparams.per_device_train_batch_size == 1
    assert hparams.optim == "adamw_8bit"

def test_load_yaml_config_missing_file():
    with pytest.raises(FileNotFoundError):
        utils.load_yaml_config("configs/does_not_exist.yaml")

def test_load_hyperparameters_missing_section(tmp_path):
    # Create a minimal config file without 'hyperparameters'
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text("setup_model:\n  model_id: 'test'\n")
    with pytest.raises(KeyError):
        utils.load_hyperparameters(str(config_path))