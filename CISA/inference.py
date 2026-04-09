import nltk
nltk.download('punkt')
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from nltk import sent_tokenize
import math, re
import argparse
# from styleformer import Styleformer
import warnings
warnings.filterwarnings("ignore")
import copy
import multiprocessing
import pickle as pkl
from openai import OpenAI
#from dotenv import load_dotenv
import os
from time import sleep
from tqdm.auto import tqdm
from natsort import natsorted
import random

import json
import pandas as pd
import re
from tqdm.auto import tqdm
from ast import literal_eval



def prompt_article(article):


    prompt = """
    For this article:

    {}

    End of article. 

    Extract the Vendor/Equipment of the Vulnerability, its name, if any CVE was assigned to it and if any CWE was a part of the Vulnerability.

    Return multiple if there are multiple but make sure the correct vulnerability gets assigned to the correct software.
    
    Return empty if it doesn't contain any of this information. Or those fields empty if it doesn't have that specific information. 

    Return the output as a list that is easy to parse. Each new entry should get a new sublist in the main list.  For e.g:
    
    Example 1: [Software: "", Vulnerability: "", CVE: "", CWE: ""] 
    
    Example 2: [[Software: "", Vulnerability: "", CVE: "", CWE: ""], [Software: "", Vulnerability: "", CVE: "", CWE: ""]]
    
    """.format(article)  # Ensure `article` has no extra spaces or newlines

    #print(prompt)
    return prompt


def get_original(articles, api_keys):
    
    
    user_input = prompt_article(articles)
    
    print(user_input)


    
    # return changed sentences in list target sentences
    
    
    client = OpenAI(
    api_key=api_keys,  # This is the default and can be omitted
)

    while True:
        try:
            response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": 'You are a helpful assistant that is an expert in extracted only named entities from articles and returning whether a relation exists between the named entities between the groups.'},
                                    {"role": "user", "content": """
                                                                  
{}
                           
                                    """.format(user_input)}
                        ],
                        temperature=0.01)

            #print(response)
            res=response.choices[0].message.content
            print(res)
            
            #sleep(1)
            break
        except Exception as e:
            sleep(1)
            print(e)
            if "This model's maximum context length is 4097 tokens." in str(e):
                return " "
                
                
    


    return res




if __name__=='__main__':
    # load_dotenv()
    # api_key = os.getenv("API_KEY")
    api_key =''

    
    
    
    #######################################################################
    
    
    dump=os.listdir('articles')
    print(len(dump))
    dump=natsorted(dump)
    text=[]
    for items in dump:
        with open('articles/{}'.format(items),'r') as f:
            file=f.read().strip()

        text.append(file)
        
    
    
    # print(text[0])

    #text=text[0:100]

    
    # print(text[0])
    
    
    
    
    collected=os.listdir('outputs')
    count=0

    for article in tqdm(text):
        cnn=[]
        # context = " ".join(article)
        #print(context)
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        # elif article==' ':
        #     with open('data_NAACL/run1/cnn/{}.pkl'.format(count), 'wb') as f:
        #         pkl.dump([' '],f)
        #     count+=1
        #     continue
      
        
        else:    
            try:    
                cnn.append(get_original(article, api_key))
                #print(cnn)
                with open('outputs/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(cnn,f)
            except Exception as e:
                print(e)
            
            
            count+=1  




   