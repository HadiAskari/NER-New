import nltk
nltk.download('punkt')

import re, os, json, pickle as pkl
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from time import sleep
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def prompt_article(article):
    prompt = """You are extracting named entities for a cybersecurity knowledge graph.

Task
- Extract ONLY the following four categories from the article text:
1) Software — benign products/packages/services/applications/libraries/platform components explicitly mentioned as relevant targets or affected items. Include version/build if present.
2) Hardware — physical chips/devices/models (e.g., CPUs, phones, routers). Include model/stepping if present.
3) Software Security Vulnerabilities — CVEs and named exploits/malware/attack campaigns that TARGET software.
4) Hardware Security Vulnerabilities — CVEs and named side-channel classes/attacks that TARGET hardware.

Core objective
- Be conservative.
- Prioritize precision over recall.
- When uncertain, exclude the item.
- Do NOT infer entities or vulnerabilities from background knowledge.
- Only extract items clearly supported by the article text.

Classification rules (very important)
- If a name refers to an attack, exploit, malware, malicious package, trojan, campaign, loader, stealer, botnet, ransomware, backdoor, or vulnerability class (e.g., "Spectre", "BlazeStealer", "SQL injection"), classify it ONLY as a Vulnerability — NEVER as Software or Hardware.
- If something could be both a legitimate tool and malware, treat it as a Vulnerability if the article discusses it as malicious.
- Only include vulnerabilities that the article states or clearly implies are affecting one of the extracted Software/Hardware items.
- Exclude generic/vague terms (e.g., "antivirus", "browser", "CPU", "server") unless a specific product/model/version is given.
- Preserve surface forms from the article where possible; do not add unsupported words.

Version handling (critical)
- If a product/package is listed with multiple versions, create a SEPARATE entity for EACH version.
- Do NOT keep multiple versions bundled into a single entity string.
- Prefer normalized per-version entities whenever the article provides a product name plus version list.
- Examples:
  - "@nx/devkit in versions: 21.5.0, 20.9.0" -> ["@nx/devkit 21.5.0", "@nx/devkit 20.9.0"]
  - "Ultralytics versions 8.3.41 and 8.3.42" -> ["Ultralytics 8.3.41", "Ultralytics 8.3.42"]
- If no version is given, keep the entity without a version.
- Include version/build if present, but only one version per entity string.

Special caution for software/package hosts and registries (critical)
- Be extremely careful with software hosts, registries, repositories, and ecosystems such as PyPI, npm, NPM, RubyGems, crates.io, Docker Hub, GitHub, GitLab, WordPress Plugin Directory, Chrome Web Store, App Store, Play Store, package repositories, or similar distribution platforms.
- If an article says a malicious package was uploaded to / distributed via / found on PyPI or npm, that does NOT mean PyPI or npm is itself the affected software.
- Do NOT extract PyPI, npm, NPM, GitHub, etc. as software merely because they hosted, distributed, indexed, or contained a malicious package.
- Only extract a host/platform/registry as software if the article explicitly says that the host/platform itself is the target, affected product, compromised service, or vulnerable system.
- A malicious package hosted on PyPI/npm should generally be treated as a software vulnerability entity (malware / malicious package), not as benign software, unless the article clearly describes it as a legitimate benign package.
- Do not confuse "present on a platform" with "affecting the platform."

Conservatism / false-positive control (critical)
- Do not extract an item unless the article gives clear evidence it belongs in one of the four categories.
- If there is any ambiguity about whether something is a benign product versus a malicious artifact, prefer classifying it as a Vulnerability only if the article frames it as malicious; otherwise exclude it.
- If there is any ambiguity about whether a product is actually affected versus merely mentioned for context, do not use that mention to justify extraction of related vulnerabilities.
- Do not treat broad ecosystem names, vendor names, or infrastructure providers as affected entities unless the article explicitly states they are affected.
- When in doubt, leave it out.

Output
- Return STRICT JSON only (no prose), with this schema and key casing exactly:

{{
  "Named_Entities": {{
    "software": [],
    "hardware": [],
    "software_vulnerabilities": [],
    "hardware_vulnerabilities": []
  }}
}}

Article:
{article}
End of article.

Example output:

{{
  "Named_Entities": {{
    "software": [
      "Microsoft Defender Antivirus 1.2",
      "Python 3.12",
      "ComfyUI Impact Pack",
      "@nx/devkit 21.5.0",
      "@nx/devkit 20.9.0"
    ],
    "hardware": ["Intel Core i7-8700K"],
    "software_vulnerabilities": ["Heartbleed", "CVE-2021-3072"],
    "hardware_vulnerabilities": ["Spectre (CVE-2017-5753)"]
  }}
}}
""".format(article=article.strip())
    return prompt


def get_original(article_text, api_key):
    user_input = prompt_article(article_text)
    client = OpenAI(api_key=api_key)

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert assistant that is an expert in extracting only the requested named entities from articles."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                temperature=1
            )
            return response.choices[0].message.content

        except Exception as e:
            sleep(2)
            if "This model's maximum context length is 4097 tokens." in str(e):
                return " "
            # for any other error, loop retries


def remove_between_tags(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_item(count, item, api_key, collected):
    """Process a single article. Returns (count, status_string)."""
    out_filename = f'{count}.pkl'

    if out_filename in collected:
        return count, 'skipped'

    url    = item.get('url', '')
    title  = item.get('title', '')
    author = item.get('author', '')
    date   = item.get('date', '')
    text   = item.get('text', '')

    context = remove_between_tags(text).strip()
    if not context:
        return count, 'empty'

    extracted = get_original(context, api_key)

    result = {
        'url':        url,
        'title':      title,
        'author':     author,
        'date':       date,
        'index':      count,   # original position in data — used to reload in order
        'raw_output': extracted
    }

    with open(f'outputs/{out_filename}', 'wb') as f:
        pkl.dump(result, f)

    return count, 'done'

API_KEY = ""
if __name__ == '__main__':
    api_key = API_KEY

    with open('thehackernews_with_URL.jsonl', 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    os.makedirs('outputs', exist_ok=True)
    collected = set(os.listdir('outputs'))

    MAX_WORKERS = 10  # lower if you hit rate limits, raise if you have headroom

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_item, count, item, api_key, collected): count
            for count, item in enumerate(data)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            count = futures[future]
            try:
                idx, status = future.result()
                if status not in ('skipped', 'empty'):
                    tqdm.write(f"[{idx}] {status}")
            except Exception as e:
                tqdm.write(f"[{count}] Error: {e}")