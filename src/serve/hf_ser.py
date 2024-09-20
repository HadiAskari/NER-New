import fire
import torch
from transformers import pipeline
import pandas as pd
from ast import literal_eval
from tqdm.auto import tqdm

from ..utils import preprocess_instance

def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    max_new_tokens: int = 2054,
):    
    df=pd.read_csv('test_dataset.csv')
    text_list=df['Text'].to_list()
    entities=[]
    entity_types=[]
    for i in df.itertuples():
        entities.append(literal_eval(i[2]))
        entity_types.append(literal_eval(i[3]))
    
    generator = pipeline('text-generation', model=model_path, torch_dtype=torch.float16, device=2)
    res=[]
    for text,entity_types_list in tqdm(zip(text_list,entity_types)):
        res_entities=[]
        set_entities=set(entity_types_list)
        for entity_type in set_entities:
            
        # try:
        #     text = input('Input text: ')
        #     entity_type = input('Input entity type: ')
        # except EOFError:
        #     text = entity_type = ''
        # if not text:
        #     print("Exit...")
        #     break
            example = {"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}
            prompt = preprocess_instance(example['conversations'])
            outputs = generator(prompt, max_length=max_new_tokens, return_full_text=False)
            #print(outputs[0]['generated_text'])
            res_entities.append(outputs[0]['generated_text'])
        res.append(res_entities)
    
    df['UniNER']=res
    
    df.to_csv('res_dataset_UniNER.csv', index=False)

if __name__ == "__main__":
    fire.Fire(main)