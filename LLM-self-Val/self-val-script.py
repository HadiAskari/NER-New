import openai
import json
from typing import List, Dict

# ---- Step 1: Load Datasets ----
def load_cve_descriptions(cve_file: str) -> Dict[str, str]:
    """Load a CVE dataset mapping CVE IDs to descriptions."""
    with open(cve_file, 'r') as f:
        cve_data = json.load(f)
    return {entry["cve_id"]: entry["description"] for entry in cve_data}


# ---- Step 2: Retrieve CVE Descriptions ----
def get_verified_description(triple: Dict[str, str], cve_db: Dict[str, str]) -> str:
    """Retrieve the verified CVE description based on CVE ID in the triple."""
    cve_id = triple.get("cve_id")
    return cve_db.get(cve_id, "CVE description not found.")


# ---- Step 3: Validate Triple Using GPT-4o ----
def validate_triple_with_gpt(triple: Dict[str, str], cve_description: str) -> str:
    """Prompt GPT-4o to validate the triple using retrieved description."""
    prompt = (
        f"Given the following CVE description:\n\n"
        f"{cve_description}\n\n"
        f"Is the following knowledge triple correct based on this description?\n"
        f"Subject: {triple['subject']}\n"
        f"Predicate: {triple['predicate']}\n"
        f"Object: {triple['object']}\n\n"
        "Answer 'Yes' or 'No' and briefly justify."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response['choices'][0]['message']['content']


# ---- Pipeline Runner ----
def run_validation_pipeline(triples_file: str, cve_file: str):
    with open(triples_file, 'r') as f:
        triples = json.load(f)

    cve_db = load_cve_descriptions(cve_file)

    for triple in triples:
        cve_description = get_verified_description(triple, cve_db)
        validation = validate_triple_with_gpt(triple, cve_description)
        print(f"Triple: {triple}")
        print(f"Validation: {validation}\n{'-'*50}")


# Example usage:
# run_validation_pipeline("triples.json", "cve_dataset.json")
