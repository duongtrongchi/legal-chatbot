import os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset

load_dotenv()


def load_dataset_from_hub(dataset_id: str, text_field: str) -> Dataset:
    if not dataset_id:
        raise ValueError("You must provide a dataset_id.")
        
    if not text_field:
        raise ValueError("You must provide a text_field name to rename.")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is missing from environment variables.")

    dataset = load_dataset(
        path=dataset_id,
        split="train",
        token=token.strip()
    ).rename_column(text_field, "text")

    return dataset

def etl_pipeline(dataset_id) -> Dataset:
    ds = load_dataset("DuongTrongChi/luatvn-split-v_0.2.0", "split", split="train")


    


