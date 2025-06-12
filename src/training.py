import os
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bfloat16_supported
from unsloth import (
    UnslothTrainer, 
    UnslothTrainingArguments
)

from src.utils import setup_model, log_hyperparameters
from src.settings import QwenHyperparameterConfig
from src.processing import load_dataset_from_hub


args = QwenHyperparameterConfig()
dataset = load_dataset_from_hub("DuongTrongChi/luatvn-split-v_0.2.0", "article").select(range(1_000))
model, tokenizer = setup_model()
log_hyperparameters(args)


trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 4096,
    dataset_num_proc = 2,
    args = UnslothTrainingArguments(**args.__dict__),
)

trainer_stats = trainer.train()

model.push_to_hub("DuongTrongChi/legal-pretrain", token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub("DuongTrongChi/legal-pretrain", token=os.getenv("HF_TOKEN"))