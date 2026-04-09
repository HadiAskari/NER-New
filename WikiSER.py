from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd 
from tqdm.auto import tqdm
from ast import literal_eval


def define():
    tokenizer = AutoTokenizer.from_pretrained("taidng/wikiser-bert-large")
    model = AutoModelForTokenClassification.from_pretrained("taidng/wikiser-bert-large")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=2)
    
    return nlp
        


def infer(nlp, text):
    
    ner_results = nlp(text)
    # print(ner_results)
    words=[]
    for k,ent in enumerate(ner_results):
        if k==0:
            words.append(ent['word'])
        elif ent['entity'].startswith('B-') and k!=0 and not ent['word'].startswith('#'):
            words.append(ent['word'])
        else:
            if ent['word'].startswith('#'):
                temp=ent['word'].strip('#')
                # print(temp)
                words[-1]=words[-1] + "" + temp
            else:
                words[-1]=words[-1] + " " + ent['word']
    # print(words)
    return words
    