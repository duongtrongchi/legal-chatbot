import os
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bfloat16_supported
from unsloth import (
    UnslothTrainer, 
    UnslothTrainingArguments
)
from transformers import DataCollatorForLanguageModeling

from src.utils import setup_model, log_hyperparameters, load_yaml_config
from src.settings import QwenHyperparameterConfig, UnslothTrainerConfig
from src.processing import load_dataset_from_hub


args = QwenHyperparameterConfig()
unsloth_trainer_config = load_yaml_config("./configs/training.yaml")["unsloth_trainer_config"]
unsloth_trainer_config = UnslothTrainerConfig(**unsloth_trainer_config)

dataset = load_dataset_from_hub("DuongTrongChi/legal-pretrain", "processed").select(range(1_000))
model, tokenizer = setup_model()
log_hyperparameters(args)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, model=model)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator=data_collator,
    dataset_text_field = unsloth_trainer_config.dataset_text_field,
    max_seq_length = unsloth_trainer_config.max_seq_length,
    dataset_num_proc = unsloth_trainer_config.dataset_num_proc,
    args = UnslothTrainingArguments(**args.__dict__),
)

trainer_stats = trainer.train()

model.push_to_hub("DuongTrongChi/legal-pretrain", token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub("DuongTrongChi/legal-pretrain", token=os.getenv("HF_TOKEN"))