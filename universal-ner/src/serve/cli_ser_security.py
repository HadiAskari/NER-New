import fire
import torch
from transformers import pipeline
import pandas as pd
from ast import literal_eval
from tqdm.auto import tqdm
from vllm import LLM
from transformers import LlamaTokenizer
from collections import defaultdict

from .inference import inference

def main(
    # model_path: str = "Universal-NER/UniNER-7B-all",
    model_path: str, #= "universal-ner/src/train/saved_models/qlora-UniNER/checkpoint-37200",
    max_new_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_input_length: int = 2048,
):      
    df=pd.read_csv('sec_data_test.csv')
    text_list=df['Text'].to_list()
    entities=[]
    entity_types=[]
    for i in df.itertuples():
        entities.append(literal_eval(i[2]))
        entity_types.append(literal_eval(i[3]))
    
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    res=[]
    for text,entity_types_list in tqdm(zip(text_list,entity_types)):
        res_entities=defaultdict(list)
        set_entities=set(entity_types_list)
        for entity_type in set_entities:
            if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
                print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")
                # continue
            
        # try:
        #     text = input('Input text: ')
        #     entity_type = input('Input entity type: ')
        # except EOFError:
        #     text = entity_type = ''
        # if not text:
        #     print("Exit...")
        #     break
            #preppend stuff
            #entity_type= "Software_Engineering_"+entity_type
            
            examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
            output = inference(llm, examples, max_new_tokens=max_new_tokens)[0]
                #print(outputs[0]['generated_text'])
            
            res_entities[entity_type].append(output)
        res.append(res_entities)
    
    df['UniNER']=res
    
    df.to_csv('UNINER-sec_data-lora.csv', index=False)

if __name__ == "__main__":
    fire.Fire(main)