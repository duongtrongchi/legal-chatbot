import os
import redis
import json
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
load_dotenv()


r = redis.Redis(host='localhost', port=6379, db=1)
ds = load_dataset("DuongTrongChi/legal-pretrain", "processed", token=os.getenv("HF_TOKEN"), num_proc=4)


def insert_data(key: int, value: dict):
    value_json = json.dumps(value)
    r.set(key, value_json)


def get_data(key: int) -> dict:
    value_json = r.get(key)
    if value_json:
        return json.loads(value_json)
    return None


# for i in tqdm(ds['train'], desc="Processing"):
#     _id = i.get('id')
#     date = i['doc_property'].get('EffectDate', None)
#     if date != None:
#         date = date.strftime("%Y-%m-%d")
#     else:
#         date = None
        
#     data = {
#         'DocIdentity': i['doc_property'].get('DocIdentity', None),
#         'DocName': i['doc_property'].get('DocName', None),
#         'EffectDate': date,
#         'OrganName': i['doc_property'].get('OrganName', None)
#     }

#     insert_data(_id, data)
