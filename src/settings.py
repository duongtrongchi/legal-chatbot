from pydantic import BaseModel
from typing import Optional
from unsloth import is_bfloat16_supported


class SetupModelConfig(BaseModel):
    model_id: str
    max_seq_length: Optional[int] = 1024
    dtype: Optional[str] = None  
    full_finetuning: bool = True


class HyperparameterConfig(BaseModel):
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 120
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
