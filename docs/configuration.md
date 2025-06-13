# Configuration & Settings

This module defines configuration classes for model setup, training hyperparameters, and trainer settings. These are used throughout the project and can be loaded from YAML files.

## Classes

### `BaseConfig`
- Inherits from `pydantic.BaseModel`.
- Utility methods:
  - `to_dict()`: Returns config as a dictionary.
  - `show()`: Prints all config fields and values.

### `SetupModelConfig`
- Fields:
  - `model_id: str` — Model identifier (e.g., Hugging Face model name)
  - `max_seq_length: Optional[int]` — Maximum sequence length (default: 1024)
  - `dtype: Optional[str]` — Data type for model weights
  - `full_finetuning: bool` — Whether to enable full finetuning (default: True)

### `HyperparameterConfig`
- Fields:
  - `per_device_train_batch_size: int` (default: 1)
  - `gradient_accumulation_steps: int` (default: 8)
  - `num_train_epochs: int` (default: 1)
  - `warmup_steps: int` (default: 10)
  - `learning_rate: float` (default: 1e-5)
  - `embedding_learning_rate: float` (default: 1e-5)
  - `fp16: bool` (default: not is_bfloat16_supported())
  - `bf16: bool` (default: is_bfloat16_supported())
  - `logging_steps: int` (default: 1)
  - `optim: str` (default: "adamw_8bit")
  - `weight_decay: float` (default: 0.01)
  - `lr_scheduler_type: str` (default: "linear")
  - `seed: int` (default: 42)
  - `output_dir: str` (default: "outputs")
  - `report_to: str` (default: "none")

### `UnslothTrainerConfig`
- Fields:
  - `dataset_text_field: str` (default: "text")
  - `max_seq_length: int` (default: 4096)
  - `dataset_num_proc: int` (default: 2)

### Model-Specific Hyperparameter Subclasses
- `LlamaHyperparameterConfig`: Overrides optimizer and adds `max_steps`.
- `QwenHyperparameterConfig`: Adjusts learning rate, warmup steps, and scheduler type.

## YAML Configuration Example
```yaml
setup_model:
  model_id: "Qwen/Qwen3-1.7B"
  max_seq_length: 4096
  dtype: null
  full_finetuning: true

hyperparameters:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 4
  warmup_steps: 10
  learning_rate: 1e-5
  embedding_learning_rate: 1e-5
  logging_steps: 1
  optim: "adamw_8bit"
  weight_decay: 0.01
  lr_scheduler_type: "linear"
  seed: 42
  output_dir: "outputs"
  report_to: "none"
  num_train_epochs: 1

unsloth_trainer_config:
  dataset_text_field: "text"
  max_seq_length: 4096
  dataset_num_proc: 2
```

## Usage Example
```python
from src.settings import QwenHyperparameterConfig
args = QwenHyperparameterConfig()
print(args.to_dict())
```