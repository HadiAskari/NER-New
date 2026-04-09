import os, json, re, pickle as pkl
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm.auto import tqdm

# ---------- CONFIG ----------
API_KEY = ""
ARTICLES_JSONL  = "thehackernews_with_URL.jsonl"
OUTPUTS_DIR     = "outputs"          # NER results: 0.pkl, 1.pkl, ...
RELATIONS_DIR   = "relation_outputs" # where we save per-article relation pkls
MODEL           = "gpt-4o-mini"
TEMPERATURE     = 1.0
MAX_WORKERS     = 10
# ----------------------------

client = OpenAI(api_key=API_KEY)


# ── helpers ──────────────────────────────────────────────────────────────────

def remove_between_tags(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def safe_json_loads(s: str):
    if not isinstance(s, str):
        raise ValueError("Expected string for JSON content.")
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def build_relation_prompt(article_text: str, entities_json: dict) -> str:
    return """
You are a cybersecurity relation extractor. Decide ONLY from the article text provided.

Goal
- Determine whether each listed vulnerability affects any of the listed software/hardware entities.
- Consider ONLY the provided article text (no outside knowledge).
- Be conservative and optimize for fewer false positives.
- If unsure, if the evidence is weak, or if the relation is only indirectly suggested, do not create a relation.

Inputs
- Article (verbatim):
{article}

- Named entities (verbatim):
{entities}

Output format
- Return STRICT JSON with exactly this schema:

{{
  "Relations": [
    {{
      "vulnerability": "",
      "affects": {{"package_name": "version"}},
      "affects_type": "software|hardware"
    }}
  ]
}}

Rules
- A vulnerability name (e.g., CVE-IDs, malware/campaign names, malicious package names) must be in the vulnerabilities lists.
- "affects" must be a JSON object containing exactly one key-value pair.
- In that key-value pair:
  - the key must be the affected software/hardware name
  - the value must be the version string if a version is present in the entity
  - otherwise the value must be ""
- The affected name/version must be derived from exactly one entity string from the software/hardware lists.
- Do NOT invent entities, vulnerabilities, names, or versions beyond the lists.
- If no relations are clearly supported by the article, return {{ "Relations": [] }}.

How to construct "affects" (critical)
- Each entity should already represent at most one version.
- If the entity string includes a version, split it into:
  - package/product name as the key
  - version as the value
- If the entity string has no version, use the full entity name as the key and "" as the value.
- Examples:
  - "Ultralytics 8.3.41" -> {{"Ultralytics": "8.3.41"}}
  - "@nx/enterprise-cloud 3.2.0" -> {{"@nx/enterprise-cloud": "3.2.0"}}
  - "ComfyUI Impact Pack" -> {{"ComfyUI Impact Pack": ""}}
  - "Intel Core i7-8700K" -> {{"Intel Core i7-8700K": ""}}

Evidence standard (critical)
- Create a relation only when the article explicitly states or clearly implies that the vulnerability affects the entity.
- Mentions of co-occurrence, proximity in text, shared sentence context, or same paragraph are NOT sufficient by themselves.
- Do not rely on background knowledge, common associations, or what is typically affected in the real world.
- If the article only says a vulnerability/package was found on, uploaded to, distributed through, or available via a platform, do NOT conclude that the platform is affected.

Special caution for software hosts and registries (critical)
- Be extremely careful with hosts/platforms/registries/ecosystems such as PyPI, npm, NPM, GitHub, GitLab, RubyGems, Docker Hub, Chrome Web Store, App Store, Play Store, package repositories, and similar services.
- If a malicious package or malware is hosted on PyPI/npm/etc., do NOT create a relation saying that the malicious package affects PyPI/npm/etc. unless the article explicitly says the platform/service itself was compromised or targeted.
- "Hosted on", "published to", "distributed via", "available on", "found in", or "uploaded to" do NOT imply "affects".

Conservatism / false-positive control (critical)
- Prefer missing a doubtful relation over adding a questionable one.
- If there is any ambiguity about whether the vulnerability targets the entity directly, do not create the relation.
- If the article describes a package as malicious but does not state what it affects, return no relation for that pair.
- Do not connect a vulnerability to broad infrastructure, vendors, or ecosystem names unless the article explicitly says they are affected.
- When in doubt, leave the relation out.

Example output:
{{
  "Relations": [
    {{
      "vulnerability": "CVE-2021-3072",
      "affects": {{"Microsoft Defender Antivirus": "1.2"}},
      "affects_type": "software"
    }},
    {{
      "vulnerability": "Spectre (CVE-2017-5753)",
      "affects": {{"Intel Core i7-8700K": ""}},
      "affects_type": "hardware"
    }}
  ]
}}
""".format(
        article=article_text.strip(),
        entities=json.dumps(entities_json, ensure_ascii=False, indent=2)
    )


# ── per-article worker ────────────────────────────────────────────────────────

def process_article(idx: int, article_text: str, ner_pkl_path: str, collected: set):
    """
    Load NER result for article `idx`, run relation extraction, save to RELATIONS_DIR.
    Returns (idx, status_string).
    """
    out_filename = f'{idx}.pkl'

    if out_filename in collected:
        return idx, 'skipped'

    # ── load NER pickle (new dict format) ────────────────────────────────────
    with open(ner_pkl_path, 'rb') as f:
        loaded = pkl.load(f)

    # support both new dict format and legacy [string] list format
    if isinstance(loaded, dict):
        raw = loaded.get('raw_output', '')
        url    = loaded.get('url', '')
        title  = loaded.get('title', '')
        author = loaded.get('author', '')
        date   = loaded.get('date', '')
    elif isinstance(loaded, list) and loaded:
        raw    = loaded[0]
        url = title = author = date = ''
    else:
        return idx, 'malformed_pkl'

    # ── parse Named_Entities ─────────────────────────────────────────────────
    try:
        entities = safe_json_loads(raw)
    except Exception:
        try:
            entities = json.loads(raw.replace("'", '"'))
        except Exception:
            return idx, 'json_parse_error'

    named    = entities.get("Named_Entities", {})
    sw       = named.get("software", [])       or []
    hw       = named.get("hardware", [])       or []
    sw_vulns = named.get("software_vulnerabilities", []) or []
    hw_vulns = named.get("hardware_vulnerabilities", []) or []

    # ── skip LLM if ALL four entity lists are empty ───────────────────────────
    if not sw and not hw and not sw_vulns and not hw_vulns:
        result = {
            'url': url, 'title': title, 'author': author,
            'date': date, 'index': idx,
            'sw_relations': [], 'hw_relations': []
        }
        with open(os.path.join(RELATIONS_DIR, out_filename), 'wb') as f:
            pkl.dump(result, f)
        return idx, 'empty_entities'

    # ── also skip LLM if no vulnerabilities (nothing to link) ────────────────
    if not sw_vulns and not hw_vulns:
        result = {
            'url': url, 'title': title, 'author': author,
            'date': date, 'index': idx,
            'sw_relations': [], 'hw_relations': []
        }
        with open(os.path.join(RELATIONS_DIR, out_filename), 'wb') as f:
            pkl.dump(result, f)
        return idx, 'no_vulns'

    # ── call LLM for relation extraction ─────────────────────────────────────
    prompt = build_relation_prompt(article_text, {
        "Named_Entities": {
            "software": sw, "hardware": hw,
            "software_vulnerabilities": sw_vulns,
            "hardware_vulnerabilities": hw_vulns
        }
    })

    relations_json = {"Relations": []}
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You extract ONLY text-supported vulnerability→affected-entity relations."},
                    {"role": "user",   "content": prompt}
                ]
            )
            content = resp.choices[0].message.content
            parsed  = safe_json_loads(content)
            if "Relations" in parsed and isinstance(parsed["Relations"], list):
                relations_json = parsed
                break
        except Exception as e:
            if "maximum context length" in str(e).lower():
                # truncate article and retry
                prompt = build_relation_prompt(article_text[:8000], {
                    "Named_Entities": {
                        "software": sw, "hardware": hw,
                        "software_vulnerabilities": sw_vulns,
                        "hardware_vulnerabilities": hw_vulns
                    }
                })
            sleep(0.25)

    # ── parse relation output into string pairs ───────────────────────────────
    sw_pairs, hw_pairs = [], []

    for rel in relations_json.get("Relations", []):
        vuln    = rel.get("vulnerability", "").strip()
        affects = rel.get("affects", {})          # dict {"package": "version"}
        atype   = rel.get("affects_type", "").strip().lower()

        if not vuln or not isinstance(affects, dict) or not affects:
            continue

        # unpack the single key-value pair
        package, version = next(iter(affects.items()))
        package = package.strip()
        version = version.strip()

        label = f"{package} {version}".strip() if version else package

        if atype == "software":
            sw_pairs.append(f"{label}, {vuln}")
        elif atype == "hardware":
            hw_pairs.append(f"{label}, {vuln}")

    result = {
        'url': url, 'title': title, 'author': author,
        'date': date, 'index': idx,
        'sw_relations': sw_pairs,
        'hw_relations': hw_pairs
    }

    with open(os.path.join(RELATIONS_DIR, out_filename), 'wb') as f:
        pkl.dump(result, f)

    return idx, 'done'


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1) load articles in original order
    with open(ARTICLES_JSONL, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    articles_text = [
        remove_between_tags(item.get('text', '')).strip()
        for item in data
    ]

    # 2) load NER outputs — key them by the index stored INSIDE the pickle
    #    so gaps (skipped empty articles) don't break alignment
    ner_pkl_by_index = {}
    for fname in os.listdir(OUTPUTS_DIR):
        if not fname.endswith('.pkl'):
            continue
        fpath = os.path.join(OUTPUTS_DIR, fname)
        try:
            with open(fpath, 'rb') as f:
                obj = pkl.load(f)
            # new dict format stores 'index'; fall back to filename stem
            if isinstance(obj, dict) and 'index' in obj:
                idx = obj['index']
            else:
                idx = int(fname.split('.')[0])
            ner_pkl_by_index[idx] = fpath
        except Exception as e:
            tqdm.write(f"Warning: could not load {fname}: {e}")

    os.makedirs(RELATIONS_DIR, exist_ok=True)
    collected = set(os.listdir(RELATIONS_DIR))

    # 3) only process articles that have a corresponding NER pkl
    valid_indices = sorted(ner_pkl_by_index.keys())
    tqdm.write(f"Articles total: {len(articles_text)} | NER outputs found: {len(valid_indices)}")

    # 4) parallel relation extraction
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_article,
                idx,
                articles_text[idx],
                ner_pkl_by_index[idx],
                collected
            ): idx
            for idx in valid_indices
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            try:
                i, status = future.result()
                if status not in ('skipped', 'empty_entities', 'no_vulns'):
                    tqdm.write(f"[{i}] {status}")
            except Exception as e:
                tqdm.write(f"[{idx}] Error: {e}")

    # 5) reload in order and export CSV
    results = []
    for fname in sorted(os.listdir(RELATIONS_DIR), key=lambda x: int(x.split('.')[0])):
        with open(os.path.join(RELATIONS_DIR, fname), 'rb') as f:
            results.append(pkl.load(f))

    import csv
    with open("software_relations_per_article.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article_index", "url", "title", "software_relations", "hardware_relations"])
        for r in results:
            w.writerow([
                r['index'],
                r.get('url', ''),
                r.get('title', ''),
                "; ".join(r['sw_relations']),
                "; ".join(r['hw_relations'])
            ])