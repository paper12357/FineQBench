import time
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("please set OPENROUTER_API_KEY environment variable or pass api_key to LLMClient")
        
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

    def call_llm(self, prompt: str, model: str = 'deepseek/deepseek-chat', max_tokens: int = 50000) -> str:
        attempt = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.8,
                    presence_penalty=0.2,
                    frequency_penalty=0.2
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                attempt += 1
                wait_time = 0.2
                print(f"[LLM CALL] LLM call failed (attempt {attempt}): {e}")

                if attempt >= 5:
                    print("[LLM CALL ERROR] Reached max retries. Giving up.")
                    raise e
                time.sleep(wait_time)

import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path

# tqdm for nicer progress bars if available (optional)
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# ---- Config ----
LOTUS_JSON = "dblotus\\lotus_football.json"
OUT_DIR = Path("dbgraph\\football")
OUT_DIR.mkdir(parents=True, exist_ok=True)
USE_CHUNKS = 5000000
BATCH_SIZE = 3
MAX_WORKERS = 10 
SEMAPHORE_LIMIT = 10
MERGE_CHUNK_PATH = OUT_DIR / "merged_chunks.json"
CHUNKS_ENT_PATH = OUT_DIR / "chunks_entities.jsonl"
CHUNKS_REL_PATH = OUT_DIR / "chunks_relations.jsonl"
KG_NODES_PATH = OUT_DIR / "kg_nodes.json"
KG_EDGES_PATH = OUT_DIR / "kg_edges.json"

llm = LLMClient()

ENTITY_REL_PROMPT_TEMPLATE = """
You are an information extraction assistant. From the following text chunk, extract:
1) Important entities, each with a brief description (1–2 sentences).
2) Relationships between entities. For each relationship, provide the source entity, target entity, relation type/phrase, and a brief description.
3) Important factual claims (optional, at most 1–3), as a list of short sentences.

Input format:
### CHUNK START
{text}
### CHUNK END

Output format (strict JSON):
{
  "entities": [
    {"id": "<entity_name>", "description": "<brief description>"},
    ...
  ],
  "relations": [
    {"source": "<entity_name>", "target": "<entity_name>", "type": "<relation type>", "description": "<brief description>"},
    ...
  ],
  "claims": [
    "<claim sentence 1>",
    ...
  ]
}

Notes:
- Entity names should use the original strings appearing in the text (preserve casing or apply reasonable normalization).
- If no claims are found, set the claims field to an empty array [].
- The output must be valid JSON and strictly follow the format above.
- Return JSON only. Do not include code blocks or any additional text.
"""

# ---- Helpers ----
def load_lotus_chunks(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(obj: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(obj: dict, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def call_entity_rel_llm(chunk_text: str, max_retries=5) -> dict:
    prompt = ENTITY_REL_PROMPT_TEMPLATE.format(text=chunk_text)
    attempt = 0
    while True:
        try:
            resp = llm.call_llm(prompt, model='deepseek/deepseek-chat', max_tokens=2000)
            import re
            m = re.search(r'\{(?:.|\n)*\}$', resp.strip())
            json_text = resp if m is None else m.group(0)
            parsed = json.loads(json_text)
            return parsed
        except Exception as e:
            attempt += 1
            print(f"[LLM EXTRACT] attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                print("[LLM EXTRACT] giving up for this chunk, returning empty structure")
                return {"entities": [], "relations": [], "claims": []}
            time.sleep(1.0)

def normalize_entity_name(name: str) -> str:
    return " ".join(name.strip().split())

# ---- Step A: load chunks ----
print("==> Step A: Load lotus chunks")
if not os.path.exists(LOTUS_JSON):
    raise FileNotFoundError(f"File not found: {LOTUS_JSON}")
chunks_by_table = load_lotus_chunks(LOTUS_JSON)
all_chunks = []
for table, paragraphs in chunks_by_table.items():
    for i, p in enumerate(paragraphs):
        chunk_id = f"{table}__{i}"
        all_chunks.append({"chunk_id": chunk_id, "table": table, "text": p})

all_chunks = all_chunks[:USE_CHUNKS]

def merge_chunks_randomly(all_chunks, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    chunks = all_chunks[:]  # shallow copy
    random.shuffle(chunks)

    merged = []
    for i in range(0, len(chunks), batch_size):
        group = chunks[i:i+batch_size]

        merged_texts = [c["text"] for c in group]
        combined_text = "\n\n".join(merged_texts)

        tables_in_group = [c.get("table", "") for c in group]
        seen = set()
        ordered_unique_tables = []
        for t in tables_in_group:
            if t not in seen:
                seen.add(t)
                ordered_unique_tables.append(t)
        combined_table = ",".join(ordered_unique_tables)

        new_id = str(len(merged) + 1)

        merged.append({
            "chunk_id": new_id,
            "table": combined_table,
            "text": combined_text
        })

    return merged

merged_chunks = merge_chunks_randomly(all_chunks, BATCH_SIZE)
save_json(merged_chunks, MERGE_CHUNK_PATH)

all_chunks = merged_chunks

# ---- Step B: Extract entities & relations per chunk (LLM) ----
print("==> Step B: Extract entities & relations per chunk via LLM")
# clear output files
for p in [CHUNKS_ENT_PATH, CHUNKS_REL_PATH]:
    if p.exists():
        p.unlink()

if TQDM:
    iterator = tqdm(all_chunks)
else:
    iterator = all_chunks

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

dict_lock = threading.Lock()
file_lock = threading.Lock()
semaphore = threading.Semaphore(SEMAPHORE_LIMIT)


chunk_entities_index = {}  # chunk_id -> entities list
chunk_relations_index = {} # chunk_id -> relations list
chunk_claims_index = {}

def process_chunk(chunk):
    cid = chunk["chunk_id"]
    print(f"[B] Processing chunk {cid} ...")
    with semaphore:
        parsed = call_entity_rel_llm(chunk["text"])

    entities = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    claims = parsed.get("claims", [])

    for e in entities:
        e["id"] = normalize_entity_name(e.get("id", ""))
        e["description"] = e.get("description", "").strip()
    for r in relations:
        r["source"] = normalize_entity_name(r.get("source", ""))
        r["target"] = normalize_entity_name(r.get("target", ""))
        r["type"] = r.get("type", "").strip()
        r["description"] = r.get("description", "").strip()

    with dict_lock:
        chunk_entities_index[cid] = entities
        chunk_relations_index[cid] = relations
        chunk_claims_index[cid] = claims

    with file_lock:
        # persist per-chunk outputs immediately
        append_jsonl({"chunk_id": cid, "entities": entities, "num_entities": len(entities)}, CHUNKS_ENT_PATH)
        append_jsonl({"chunk_id": cid, "relations": relations, "num_relations": len(relations)}, CHUNKS_REL_PATH)

    # tiny sleep to avoid hammering
    time.sleep(0.01)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(process_chunk, chunk) for chunk in iterator]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        try:
            fut.result()
        except Exception as e:
            print("Error processing chunk:", e)

print(f"[B] Done. Extracted entities/relations for {len(chunk_entities_index)} chunks.")

# ---- Step C: Aggregate into Knowledge Graph ----
print("==> Step C: Aggregate extracted entities/relations -> Knowledge Graph")
# Use exact string matching for entity reconciliation (as you requested)
nodes = {}  # name -> {name, descriptions: [..], count}
edges = {}  # (src, tgt, type) -> {source, target, type, descriptions:[], weight}

for cid, ents in chunk_entities_index.items():
    for e in ents:
        name = e["id"] or e.get("name") or ""
        if not name: continue
        if name not in nodes:
            nodes[name] = {"name": name, "descriptions": [], "count": 0}
        if e.get("description"):
            nodes[name]["descriptions"].append(e["description"])
        nodes[name]["count"] += 1

for cid, rels in chunk_relations_index.items():
    for r in rels:
        src = r.get("source"); tgt = r.get("target"); rtype = r.get("type") or "related_to"
        if not src or not tgt: continue
        key = (src, tgt, rtype)
        if key not in edges:
            edges[key] = {"source": src, "target": tgt, "type": rtype, "descriptions": [], "weight": 0}
        if r.get("description"):
            edges[key]["descriptions"].append(r["description"])
        edges[key]["weight"] += 1

# compress node descriptions (keep unique)
for n in nodes.values():
    n["descriptions"] = list(dict.fromkeys(n["descriptions"]))  # preserve order unique
for e in edges.values():
    e["descriptions"] = list(dict.fromkeys(e["descriptions"]))

# Save KG as node/edge lists
save_json({"nodes": list(nodes.values())}, KG_NODES_PATH)
save_json({"edges": list(edges.values())}, KG_EDGES_PATH)
print(f"[C] KG nodes saved to {KG_NODES_PATH} ({len(nodes)} nodes).")
print(f"[C] KG edges saved to {KG_EDGES_PATH} ({len(edges)} edges).")