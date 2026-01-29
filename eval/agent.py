from abc import ABC
from eval.answer_type import ANSWER_TYPE
import json
import ast
from rouge_score import rouge_scorer
from tools.openrouter_api import LLMClient

class BaseAgent(ABC):
    def __init__(self, answer_type: ANSWER_TYPE):
        self.answer_type = answer_type

class RougeAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANSWER_TYPE.ROUGE)
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"],
            use_stemmer=True
        )

    def _normalize_to_text(self, x):
        if x is None:
            return ""

        if isinstance(x, list):
            return " ".join(str(i) for i in x)

        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return " ".join(str(i) for i in parsed)
            except Exception:
                pass
            return x

        return str(x)

    def evaluate(self, question: str, reference_answer, answer):
        ref_text = self._normalize_to_text(reference_answer)
        ans_text = self._normalize_to_text(answer)

        if not ref_text or not ans_text:
            return {
                "score": 0,
                "detail": {
                    "rouge1": 0.0,
                    "rougeL": 0.0,
                    "reason": "Empty reference or answer."
                }
            }

        scores = self.scorer.score(ref_text, ans_text)

        r1 = scores["rouge1"].fmeasure
        rl = scores["rougeL"].fmeasure
        avg = (r1 + rl) / 2

        return {
            "score": int(avg * 100),
            "detail": {
                "rouge1": round(r1, 4),
                "rougeL": round(rl, 4),
                "average": round(avg, 4)
            }
        }
    
class SubtaskCompletionAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANSWER_TYPE.SUBTASK_COMP)

    def _flatten_to_text(self, x):
        """
        Recursively flatten arbitrary execution logs (dict / list / str / others)
        into a single lowercase string.
        """
        if x is None:
            return ""

        if isinstance(x, dict):
            return " ".join(
                self._flatten_to_text(k) + " " + self._flatten_to_text(v)
                for k, v in x.items()
            )

        if isinstance(x, (list, tuple, set)):
            return " ".join(self._flatten_to_text(i) for i in x)

        return str(x)

    def evaluate(self, reference_subtasks, execution_log):
        """
        reference_subtasks: List[str]
        execution_log: arbitrary structured execution log
        """

        if not reference_subtasks:
            return {
                "score": 100,
                "detail": {
                    "recall": 1.0,
                    "matched": [],
                    "missing": []
                }
            }

        log_text = self._flatten_to_text(execution_log).lower()

        matched = []
        missing = []

        for ref in reference_subtasks:
            ref_norm = str(ref).strip().lower()
            if not ref_norm:
                continue

            if ref_norm in log_text:
                matched.append(ref)
            else:
                missing.append(ref)

        recall = len(matched) / len(reference_subtasks)

        return {
            "score": int(recall * 100),
            "detail": {
                "recall": round(recall, 4),
                "matched": matched,
                "missing": missing
            }
        }

class ToolUsageAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANSWER_TYPE.TOOL_CALL)
        self.llm_client = LLMClient()

    def _flatten_logs(self, logs):
        """
        Flatten arbitrary tool call logs into a single text string.
        """
        if logs is None:
            return ""

        if isinstance(logs, dict):
            return " ".join(
                self._flatten_logs(k) + " " + self._flatten_logs(v)
                for k, v in logs.items()
            )

        if isinstance(logs, (list, tuple)):
            return " ".join(self._flatten_logs(i) for i in logs)

        return str(logs)

    def _judge_call_present(self, essential_call, log_text):
        """
        Use LLM to judge whether an essential tool call
        is semantically present in the execution logs.
        """
        prompt = f"""
            You are an evaluator for tool usage.

            Essential tool call (gold requirement):
            "{essential_call}"

            Tool invocation logs:
            "{log_text}"

            Question:
            Has the intent of the essential tool call been correctly fulfilled
            by at least one tool invocation in the logs?

            Answer with only YES or NO.
        """
        resp = self.llm_client.call_llm(prompt).strip().upper()
        return resp.startswith("YES")

    def evaluate(self, essential_tool_calls, tool_logs):
        """
        essential_tool_calls: List[str]
        tool_logs: List[dict] (arbitrary structured tool invocation records)
        """

        if not essential_tool_calls:
            return {
                "score": 100,
                "detail": {
                    "recall": 1.0,
                    "matched": [],
                    "missing": []
                }
            }

        log_text = self._flatten_logs(tool_logs)

        matched = []
        missing = []

        for call in essential_tool_calls:
            ok = self._judge_call_present(call, log_text)
            if ok:
                matched.append(call)
            else:
                missing.append(call)

        recall = len(matched) / len(essential_tool_calls)

        return {
            "score": int(recall * 100),
            "detail": {
                "recall": round(recall, 4),
                "matched": matched,
                "missing": missing
            }
        }


class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANSWER_TYPE.REPORT)
        self.llm_client = LLMClient()
    
    def evaluate(self, question: str, reference_answer: str, answer: str,) -> list:

        prompt = f"""
            You are an expert evaluator for analytical and report-style question answering tasks.
    
            Your task is to evaluate how well the **Model Answer** matches the **Reference Answer**, and output a JSON object.
    
            You must assess the answer based on the following 4 equally important criteria:
    
            1. **Factual Accuracy (0–30)** – Statements should be factually correct and consistent with the reference. Penalize hallucinations or errors.
            2. **Semantic Alignment (0–20)** – The meaning and intent should match the reference. Rewording is allowed but the semantics must stay aligned.
            3. **Completeness (0–30)** – The answer should contain all major points present in the reference. Missing critical elements reduces the score.
            4. **Expression Quality (0–20)** – Writing should be clear, concise, coherent, and appropriate for an analytical/report tone.
    
            ---
    
            Question:
            {question}
    
            Reference Answer:
            {reference_answer}
    
            Model Answer:
            {answer}
    
            ---
    
            Return **only** a JSON object with the following fields:
    
            - "score": an integer between 0 and 100 representing the total score across all criteria.
            - "reason": a concise explanation (1–3 sentences) summarizing why the score was given.
    
            Do not include anything outside the JSON object. No code blocks, no extra text.
        """
        while True:
            try:
                response = self.llm_client.call_llm(prompt)
                response_json = json.loads(response)
                score = int(response_json.get("score", 0))
                score = max(0, min(score, 100))
                reason = response_json.get("reason", "")
                break
            except Exception as e:
                print(f"[ERROR] Evaluation error: {e}. Retrying...")
       
        return {"score": score, "reason": reason}
    

class ListAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANSWER_TYPE.LIST)
        self.llm_client = LLMClient()
    
    def evaluate(self, question: str, reference_answer: str, answer: str) -> int:
        try:
            answer_list = ast.literal_eval(answer)

            reference_set = set(reference_answer)
            answer_set = set(answer_list)

            if not reference_set:
                if not answer_set:
                    return {"score": 100, "reason": "Both reference and answer are empty lists."}
                else:
                    return {"score": 0, "reason": "Reference list is empty but answer list is not."}
            if not answer_set:
                return {"score": 0, "reason": "Answer list is empty but reference list is not."}

            true_positives = len(reference_set.intersection(answer_set))
            precision = true_positives / len(answer_set)
            recall = true_positives / len(reference_set)
            if precision + recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            score = int(f1_score * 100)
            return {"score": score, 
                    "reason": f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}"}
        
        except Exception as e:
            print(f"[ERROR] List evaluation error: {e}. Falling back to REPORT evaluation.")
            rpa = ReportAgent()
            reference_answer = ', '.join(reference_answer)
            return rpa.evaluate(question, reference_answer, answer)


class StepAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANSWER_TYPE.STEP)
        self.llm_client = LLMClient()
    
    def evaluate(self, question: list, reference_answer: list, answer: str,
                 decompose: list) -> dict:
        question_main_str = question[0]

        steps_gt = question[1:]
        steps_gt_str = json.dumps(steps_gt, ensure_ascii=False, indent=2)

        answer_main_str = reference_answer[-1]

        predicted_steps = decompose
        predicted_steps_str = json.dumps(predicted_steps, ensure_ascii=False, indent=2)

        predicted_answer_str = answer

        prompt = f"""
            You are an expert evaluator for **multi-step problem solving systems**.

            Your task is to evaluate a model's performance from **two independent aspects**:

            ---

            ## Aspect 1: Final Answer Correctness (score1)

            Evaluate ONLY whether the **predicted final answer** is correct.

            - Compare the predicted answer with the reference answer.
            - Focus on factual correctness, numerical accuracy, or logical equivalence.
            - Ignore the reasoning steps completely.
            - If the answer is partially correct or approximately correct, give partial credit.

            Score range: 0–100

            ---

            ## Aspect 2: Step Decomposition Quality (score2)

            Evaluate ONLY the **quality of the predicted step decomposition**.

            - Judge whether the predicted steps form a **reasonable, logical, and executable plan**
              to solve the main question.
            - You MAY refer to the reference step decomposition for guidance.
            - The predicted steps do NOT need to match the reference steps exactly.
            - If the decomposition is different but still logically valid and sufficient, it can receive a high score.
            - Be strict about logical soundness, completeness, and executability. If there are flaws, reduce the score accordingly.
            - Do NOT consider whether the final answer is correct.

            Score range: 0–100

            ---

            ## Main Question
            {question_main_str}

            ---

            ## Reference Final Answer
            {answer_main_str}

            ---

            ## Predicted Final Answer
            {predicted_answer_str}

            ---

            ## Reference Step Decomposition
            {steps_gt_str}

            ---

            ## Predicted Step Decomposition
            {predicted_steps_str}

            ---

            ## Output requirements (STRICT)

            Return **only** a JSON object with the following fields:

            - "score1": integer between 0 and 100  
            - "score2": integer between 0 and 100  

            - "reason1": explanation for the final answer score  
              - Explain whether the predicted answer is correct, partially correct, or incorrect.

            - "reason2": explanation for the step decomposition score  
              - Evaluate logical soundness, completeness, and executability of the steps.
              - Do NOT mention final answer correctness here.

            Do NOT include anything outside the JSON object.
            No markdown, no code blocks, no extra text.
        """

        while True:
            try:
                response = self.llm_client.call_llm(prompt)
                response_json = json.loads(response)

                score_answer = response_json.get("score1", None)
                score_decomp = response_json.get("score2", None)
                reason_answer = response_json.get("reason1", "")
                reason_decomp = response_json.get("reason2", "")

                if not isinstance(score_answer, int) or not isinstance(score_decomp, int):
                    raise ValueError("Scores must be integers")

                score_answer = max(0, min(score_answer, 100))
                score_decomp = max(0, min(score_decomp, 100))

                break

            except Exception as e:
                print(f"[ERROR] StepAgent2 evaluation error: {e}. Retrying...")

        return {
            "score_answer": score_answer,
            "score_decomp": score_decomp,
            "reason_answer": reason_answer,
            "reason_decomp": reason_decomp
        }