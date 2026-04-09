import os
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from time import sleep

if not os.path.exists('articles'):
    os.makedirs('articles')

with open('advisories.csv') as f:
    advisories = f.read().split('\n')

for advisory in tqdm(advisories):
    page, url = advisory.split(',')

    if url.endswith('/'):
        url = url[:-1]
    _id = url.split('/')[-1]
    outfile = f'articles/{_id}'

    if os.path.exists(outfile):
        continue

    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', {'class':'l-page-section__content'})
    with open(outfile, 'w') as f:
        f.write(content.text)

    sleep(0.5)