import sys
import os
import json
import jsonlines
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from query_generation_agents.query_type import QUERY_TYPE
from query_generation_agents.factory import AgentFactory
from query_generation_agents.agent import *
from config import USE_DATASET

TEMPLATE_FILE = "example/example_template.jsonl"
TYPE = QUERY_TYPE.LOGIC_MULTIHOP
OUTPUT_FILE = "example/example_query.jsonl"
N = 5
USE = "football"

DATASET_DIR = USE_DATASET[USE]["dataset_dir"]
DATASET = USE_DATASET[USE]["dataset"]
TOOLS = USE_DATASET[USE]["tools"]

def main():
    templates = []
    with jsonlines.open(TEMPLATE_FILE) as reader:
        for obj in reader:
            templates.append(obj)

    agent = AgentFactory.create_agent(TYPE)

    for i, template in enumerate(templates, 1):
        print(f"[INFO] Processing template {i}/{len(templates)}: {template}")

        qa_pairs = agent.generate(
            template=template,
            dataset=DATASET,
            dataset_dir=DATASET_DIR,
            n=N,
            tools=TOOLS
        )

        with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
            for q, a in qa_pairs:
                out.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")

        print(f"[INFO] Saved {len(qa_pairs)} QA pairs → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
