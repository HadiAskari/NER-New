
from transformers import pipeline

from tqdm.auto import tqdm
import pickle as pkl
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from ast import literal_eval



def generate_res(chatbot,text,software,hardware,vulnerability):
    
    results=[]
    for desc,soft,hard,vul in tqdm(zip(text,software,hardware,vulnerability)):
        # print(desc)
        # print(soft)
        # print(hard)
        # print(vul)
        messages = [
        {"role": "system", "content": f"""
    You are an excellent content annotator. You will be provided with detailed text information of a shoppable product on some eCommerce website.
    Please provide more details about the product in terms of google product category, brand, color, material and gender.
    The output should be in the following format and only this:

    Google Product Category: Your response
    Brand: Your response
    Color: Your response
    Material: Your response
    Gender: Your response

    """.strip()
    },
        {"role": "user", "content": "Here is the textual description of the shoppable product"},
    ]
    #     messages = [
    #     {"role": "system", "content": f"""
    # You are an excellent content relation extractor. You will be provided with detailed text information a news article, a list of Software named entities in that text, a list of Hardware entities in that text, and a list of cyber-security vulnerabilities in that text. 
    # For each Software or Hardware Entity, please answer Yes or No, depending on the text, whether that entity has the Vulnerabilities in the Vulnerability list or not.
    # If either the Software or Hardware lists are empty then you can ignore that list.
    # The output should be in the following format and only this:

    # Software 1 - Vulnerability 1: Your Answer
    # Software 2 - Vulnerability 1: Your Answer
    # Hardware 1 - Vulnerability 1: Your Answer
    # Software 1 - Vulnerability 2: Your Answer

    # """.strip()
    # },
    #     {"role": "user", "content": "Here is the textual description of the article:" }, #Description: {}. Here is the Software list: {}. Here is the Hardware list: {}. Here is the Vulnerability list: {}".format(str(desc),str(soft),str(hard),str(vul))
    # ]
        # with open('see_new.txt', 'w') as f:
        #     f.write(str(messages))
        # break
        
        res=chatbot(messages)
        # print(res)
        results.append(res[0]['generated_text'][2]['content'])
        print(res[0]['generated_text'][2]['content'])
        
    
    return results

def return_list(df,key):
    item_list=[]

    for items in df.itertuples():
        #print(items[key])
        item_list.append(literal_eval(items[key]))
        
    return item_list


if __name__=='__main__':
    
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )


    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            #load_in_4bit=True,
            #quantization_config=bnb_config,
            #torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    chatbot = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        #torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_new_tokens=1024,
        temperature=0.001
    )
    
    df=pd.read_csv('Hacker_100.csv')
    text=return_list(df,1)
    software=return_list(df,2)
    hardware=return_list(df,3)
    vulnerability=return_list(df,4)

    output=generate_res(chatbot,text,software,hardware,vulnerability)
    # df['Llama3']=output
    # df.to_csv('Subsampled_eval_dataset_Llama3-8b.csv', index=False)
   #print(output)
    # df['RE']=output
    # df.to_csv('Hacker_100_RE.csv',index=False)


    