import pandas as pd
import pickle as pkl
import os
import json
from natsort import natsorted
import re
from tqdm.auto import tqdm
from ast import literal_eval
from collections import defaultdict

import spacy

# Load the SpaCy language model (replace 'en_core_web_sm' with a larger model if needed)
nlp = spacy.load('en_core_web_sm')

with open('hackerNews_updated.json', 'r') as f:
    file=json.load(f)
    
def remove_between_tags(text):
    # Remove everything between < and > including the symbols
    text = re.sub(r'<.*?>', '', text)
    # Remove any extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for items in tqdm(file.keys()):
    content_list=file[items]['content']
    temp=[]
    for sentences in content_list:
        fixed=remove_between_tags(sentences)
        if fixed is not None:
            temp.append(remove_between_tags(sentences))
    file[items]['fixed_content']=temp


results=os.listdir('outputs')

results=natsorted(results)

pkls_list=[]
for name in results:
    with open('outputs/{}'.format(name), 'rb') as f:
        pkls_list.append(pkl.load(f))
        
pased_data_list=[]
for k,items in enumerate(pkls_list):
    try:
        raw_json_str=items[0]
        cleaned_json_str = raw_json_str.strip('```json\n').strip('\n```')
        parsed_data = json.loads(cleaned_json_str)
        pased_data_list.append(parsed_data)
    except:
        pased_data_list.append({})
        print(k)
        print(cleaned_json_str)

software_per_article=[]
hardware_per_article=[]
software_vulnerabilities_per_article=[]
hardware_vulnerabilities_per_article=[]
relations_per_article=[]
for items in pased_data_list:
    try:
        software_per_article.append(items['Named_Entities']['software'])
        hardware_per_article.append(items['Named_Entities']['hardware'])
        software_vulnerabilities_per_article.append(items['Named_Entities']['software_vulnerabilities'])
        hardware_vulnerabilities_per_article.append(items['Named_Entities']['hardware_vulnerabilities'])
        relations_per_article.append(items['Relations'])
    except:
        # print(items)
        # try:
        #     software_per_article.append(items['Named_Entities']['software'])
        #     hardware_per_article.append(items['Named_Entities']['hardware'])
        #     software_vulnerabilities_per_article.append(items['Named_Entities']['software_security_vulnerabilities'])
        #     hardware_vulnerabilities_per_article.append(items['Named_Entities']['hardware_security_vulnerabilities'])
        #     relations_per_article.append(items['Relations'])
        # except:
            software_per_article.append([])
            hardware_per_article.append([])
            software_vulnerabilities_per_article.append([])
            hardware_vulnerabilities_per_article.append([])
            relations_per_article.append([])

def extract_sentences_with_entity(article, entity_name):
    """
    Extracts all sentences from an article containing the specified named entity.
    
    :param article: str, the text of the article
    :param entity_name: str, the named entity to search for
    :return: list of sentences containing the named entity
    """
    # Process the article using SpaCy
    doc = nlp(article)
    
    # Tokenize the article into sentences
    sentences = list(doc.sents)
    
    # Extract sentences containing the named entity
    sentences_with_entity = [
        sent.text for sent in sentences if entity_name.lower() in sent.text.lower()
    ]
    
    return sentences_with_entity

text_file=[]
count=0
for items in tqdm(file.keys()):
    count+=1
    text_file.append(" ".join(file[items]['fixed_content']))
    

# software_entity_sentences=defaultdict(list)
# for software, article in tqdm(zip(software_per_article, text_file)):
#     for items in software:
#         sent_list=extract_sentences_with_entity(article,items)
#         if len(sent_list)!=0:
#             for sent in sent_list:
#                 software_entity_sentences[items].append(sent)
    
# with open('software_entity_sentences', 'wb') as f:
#     pkl.dump(software_entity_sentences,f)


hardware_entity_sentences=defaultdict(list)
for hardware, article in tqdm(zip(hardware_per_article, text_file)):
    for items in hardware:
        sent_list=extract_sentences_with_entity(article,items)
        if len(sent_list)!=0:
            for sent in sent_list:
                hardware_entity_sentences[items].append(sent)

with open('hardware_entity_sentences', 'wb') as f:
    pkl.dump(hardware_entity_sentences,f)

software_vulnerability_entity_sentences=defaultdict(list)
for software_vulnerability, article in tqdm(zip(software_vulnerabilities_per_article, text_file)):
    for items in software_vulnerability:
        sent_list=extract_sentences_with_entity(article,items)
        if len(sent_list)!=0:
            for sent in sent_list:
                software_vulnerability_entity_sentences[items].append(sent)

with open('software_vulnerability_entity_sentences', 'wb') as f:
    pkl.dump(software_vulnerability_entity_sentences,f)


hardware_vulnerability_entity_sentences=defaultdict(list)
for hardware_vulnerability, article in tqdm(zip(hardware_vulnerabilities_per_article, text_file)):
    for items in hardware_vulnerability:
        sent_list=extract_sentences_with_entity(article,items)
        if len(sent_list)!=0:
            for sent in sent_list:
                hardware_vulnerability_entity_sentences[items].append(sent)
                

with open('hardware_vulnerability_entity_sentences', 'wb') as f:
    pkl.dump(hardware_vulnerability_entity_sentences,f)