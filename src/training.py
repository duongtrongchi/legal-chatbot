import os
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

import comet_ml
from comet_ml import Experiment

from unsloth import is_bfloat16_supported
from unsloth import (
    UnslothTrainer, 
    UnslothTrainingArguments
)
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from src.utils import setup_model, log_hyperparameters, load_yaml_config
from src.settings import QwenHyperparameterConfig, UnslothTrainerConfig
# from src.processing import load_dataset_from_hub


comet_ml.login(project_name="legal-chatbot", api_key=os.getenv("COMET_API_KEY"))

os.environ["COMET_LOG_ASSETS"] = "True"

args_config = load_yaml_config("./configs/training.yaml")["hyperparameters"]
args = QwenHyperparameterConfig(**args_config)

unsloth_trainer_config = load_yaml_config("./configs/training.yaml")["unsloth_trainer_config"]
unsloth_trainer_config = UnslothTrainerConfig(**unsloth_trainer_config)

dataset = load_dataset("DuongTrongChi/legal-pretrain", "processed", token=os.getenv("HF_TOKEN"), split="train").select(range(30_000))
model, tokenizer = setup_model()
log_hyperparameters(args)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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