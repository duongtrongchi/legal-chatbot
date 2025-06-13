# ========================================== V2 =======================================================================
from pydantic import BaseModel
from typing import Optional
from unsloth import is_bfloat16_supported


class BaseConfig(BaseModel):
    """Base configuration with utility methods."""

    def to_dict(self):
        return self.model_dump()

    def show(self):
        for key, value in self.model_dump().items():
            print(f"{key}: {value}")


class SetupModelConfig(BaseConfig):
    """Model-specific configuration."""
    model_id: str
    max_seq_length: Optional[int] = 1024
    dtype: Optional[str] = None
    full_finetuning: bool = True


class HyperparameterConfig(BaseConfig):
    """Training hyperparameters."""
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    warmup_steps: int = 10
    learning_rate: float = 1e-5
    embedding_learning_rate: float = 1e-5
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 42
    output_dir: str = "outputs"
    report_to: str = "none"


class UnslothTrainerConfig(BaseConfig):
    dataset_text_field: str = "text"
    max_seq_length: int = 4096
    dataset_num_proc: int = 2


# === Optional: Subclassing for specific models ===

class LlamaHyperparameterConfig(HyperparameterConfig):
    """Overrides for LLaMA-based models."""
    optim: str = "adamw_torch"
    max_steps: int = 200


class QwenHyperparameterConfig(HyperparameterConfig):
    """Overrides for Qwen models."""
    learning_rate: float = 2e-5
    warmup_steps: int = 20
    lr_scheduler_type: str = "cosine"
