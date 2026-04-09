import requests
import re
from time import sleep
from tqdm.auto import tqdm

base_url = 'https://www.cisa.gov'
with open('pages_completed') as f:
    pages_completed = int(f.read())

advisories_csv = open('advisories.csv', 'a')

for page in tqdm(range(pages_completed, 805)):
    url = base_url + f"/news-events/cybersecurity-advisories?page={page}"
    r = requests.get(url)
    html = r.text
    hrefs = re.findall(r'href="(/news-events/ics-.*?)"', html)
    for href in hrefs:
        final_url = base_url + href
        advisories_csv.write(f'{page},{final_url}\n')

    with open('pages_completed', 'w') as f:
        f.write(str(page))

    sleep(0.5)