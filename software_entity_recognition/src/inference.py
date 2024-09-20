from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd 
from tqdm.auto import tqdm
from ast import literal_eval

tokenizer = AutoTokenizer.from_pretrained("taidng/wikiser-bert-large")
model = AutoModelForTokenClassification.from_pretrained("taidng/wikiser-bert-large")

df=pd.read_csv('test_dataset.csv')
text_list=df['Text'].to_list()
entities=[]
entity_types=[]
for i in df.itertuples():
    entities.append(literal_eval(i[2]))
    entity_types.append(literal_eval(i[3]))

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

res=[]
for text in tqdm(text_list):
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
    res.append(words)
    

df['wikiSER']=res

df.to_csv('res_dataset_wikiSER.csv', index=False)