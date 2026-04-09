import json
import pandas as pd
import re
from tqdm.auto import tqdm
from ast import literal_eval
from collections import defaultdict


import pickle as pkl

with open('unique_relations.pkl', 'rb') as f:
    overal_relations=pkl.load(f)
    
with open('cve.json', 'r') as f:
    cves=json.load(f)

   
match_counts = defaultdict(list)

for item in tqdm(overal_relations):
    vals=item.split(', ')
    # print(vals)
    if vals[1].startswith('CVE'):
        continue
    elif 'vulnerabilities' in vals[1]:
        continue
    else:
        needle=vals[1]
        match_found=False
        for i in range(len(cves)):
            haystack=cves[i]['description']['description_data'][0]['value']
            if re.search(re.escape(needle), haystack, re.IGNORECASE):
                match_found=True
                match_counts[needle].append(cves[i]['identifier'])
                #print(f"Match found: {needle} in CVE {cves[i]['identifier']}")
            
        if match_found==False:
            match_counts[needle].append('No Match')


with open('match_counts.pkl', 'wb') as f:
    pkl.dump(match_counts, f)

                