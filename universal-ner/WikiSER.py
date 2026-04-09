from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModel
from transformers import pipeline
import pandas as pd 
from tqdm.auto import tqdm
from ast import literal_eval
import torch
from collections import OrderedDict


def define():
    # tokenizer = AutoTokenizer.from_pretrained("hadiaskari98/Vulnerability_NER_New")
    # #model = AutoModelForTokenClassification.from_pretrained("taidng/wikiser-bert-large")
    # model = AutoModelForTokenClassification.from_pretrained("hadiaskari98/Vulnerability_NER_New")
    # # for name, param in model.named_parameters():
    # #     #print(name)
    # weights=torch.load('/nas02/Hadi/NSF-NER/WikiSER/training/model/model_epoch_9_vulnerability-final.pt')
    # new_weights=OrderedDict()
    # for k,v in weights.items():
    #     k=k.replace('models.0.model.','bert.')
    #     k=k.replace('models.0.','')
    #     if k in ["bert.pooler.dense.weight", "bert.pooler.dense.bias",'bert.embeddings.position_ids']:
    #         continue
    #     new_weights[k]=v
    # model.load_state_dict(new_weights)
    
    # # Save the modified model
    # save_path = "fixed_vulnerability.pt"
    # torch.save(model.state_dict(), save_path)

    
    # nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=2)
    #print('Yes')
    # nlp=[]
    tokenizer = AutoTokenizer.from_pretrained("hadiaskari98/Vulnerability_NER_prod")
    model = AutoModelForTokenClassification.from_pretrained("hadiaskari98/Vulnerability_NER_prod")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=2)
    
    return nlp
        


def infer(nlp, text):
    
    ner_results = nlp(text)
    #print(ner_results)
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
    
if __name__=='__main__':
    lol=define()
    res=infer(lol,'I am working with Hash Table')
    print(res)


    # tokenizer = AutoTokenizer.from_pretrained("hadiaskari98/Software_NER_prod")
    # model = AutoModelForTokenClassification.from_pretrained("hadiaskari98/Software_NER_prod")

    # nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    # example = "Windows XP is an example of an operating system"

    # ner_results = nlp(example)
    # print(ner_results)


