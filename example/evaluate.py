import random
import sys
import os
import json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_agents.agent_type import AGENT_TYPE
from data_agents.factory import AgentFactory
from eval.answer_type import ANSWER_TYPE
from eval.factory import EvaluateFactory
from config import USE_DATASET, USE_QUERY, USE_LLM

#--------------------#
llm_name = "ds"
collection = "football"
query_type = "RTI"

agent_type = AGENT_TYPE.PLANNING
#TOOLS = ["db", "lotus", "graph"]
TOOLS = ["db"]
llm_model = USE_LLM[llm_name]
query_file = USE_QUERY[collection][query_type]

agent_name = agent_type.value
answer_type = ANSWER_TYPE.REPORT
DATASET = USE_DATASET[collection]["dataset"]
DATASET_DIR = USE_DATASET[collection]["dataset_dir"]
#--------------------#

def main():
    agent = AgentFactory.create_agent(agent_type, dataset=DATASET, dataset_dir=DATASET_DIR, 
                                      tools=TOOLS, llm_model=llm_model)

    eval = EvaluateFactory.create_agent(answer_type)

    with open(query_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line.strip()) for line in f if line.strip()]    
    random.shuffle(queries)

    queries = queries[:1]
    
    all_score = []
    all_questions = []

    log_path = os.path.join("eval_logs", f"eval_{collection}_{query_type}_{agent_name}_{llm_name}.json")
    os.makedirs("eval_logs", exist_ok=True)
    log = []

    for i, qa in enumerate(queries, 1):
        question = qa['question']
        ground_truth = qa['answer']

        print(f"[EVAL] Processing query {i}/{len(queries)}: {question}")

        while True:
            try:
                result = agent.handle_query(question)
                break
            except Exception as e:
                print(f"[EVAL ERROR] Agent failed to handle query: {e}, retrying...")
        predicted_answer = result.get('answer', '')
        token_usage = result.get('token_usage', {})

        print(f"[EVAL] Evaluating predicted answer...")

        eval_result = eval.evaluate(question, ground_truth, predicted_answer)
        score = eval_result.get('score')

        if isinstance(score, list):
            step_score = score
            score = sum(score) / len(score) if len(score) > 0 else 0.0
            score = int(score)
        else:
            step_score = None

        reason = eval_result.get('reason', '')

        print(f"[EVAL] Query {i} Score: {score:.2f}")
        all_score.append(score)
        all_questions.append(question)

        log_entry = {
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "score": score,
            "step_score": step_score,
            "reason": reason,
            "token_usage": token_usage,
            "details": result,
        }
        if step_score is None:
            log_entry.pop("step_score")

        log.append(log_entry)


    avg_score = sum(all_score) / len(all_score) if all_score else 0.0
    print(f"[EVAL] Average Score over {len(all_score)} queries: {avg_score:.2f}")

    with open(log_path, "w", encoding="utf-8") as out:
        out.write(json.dumps({
            "average_score": avg_score,
            "collection": collection,
            "query_type": query_type,
            "agent_name": agent_name,
            "llm_model": llm_model,
            "num_queries": len(all_score),
            "all_questions": all_questions,
            "all_scores": all_score,
            "detailed_log": log
        }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
