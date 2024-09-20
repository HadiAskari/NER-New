import fire
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
from transformers import pipeline
import pandas as pd
from ast import literal_eval
from tqdm.auto import tqdm
from vllm import LLM
from transformers import LlamaTokenizer
from collections import defaultdict
from WikiSER import define, infer

from .inference import inference




def main(
    #model_path: str = "Universal-NER/UniNER-7B-all",
    model_path: str,
    max_new_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_input_length: int = 2048,
):      
    df=pd.read_csv('selected_test_dataset.csv')
    text_list=df['Text'].to_list()
    entities=[]
    entity_types=[]
    for i in df.itertuples():
        entities.append(literal_eval(i[2]))
        entity_types.append(literal_eval(i[3]))
    
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    nlp=define()
    res=[]
    for text,entity_types_list in tqdm(zip(text_list,entity_types)):
        res_entities=[]
        set_entities=set(entity_types_list)
        for entity_type in set_entities:
            if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
                print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")

            #preppend stuff
            #entity_type= "Software_Engineering_"+entity_type
            
            examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
            output = inference(llm, examples, max_new_tokens=max_new_tokens)[0]
                #print(outputs[0]['generated_text'])
            
            if output:
                try:
                    res_entities.extend(literal_eval(output))
                except Exception as e:
                    print(e)
                    res_entities.extend(output)
                
        res_entities.extend(infer(nlp,text))
        #print(res_entities)
        res.append(res_entities)
    
    df['Combined']=res
    
    df.to_csv('selected_dataset_UNINER-all_combined_finetune.csv', index=False)

if __name__ == "__main__":
    fire.Fire(main)