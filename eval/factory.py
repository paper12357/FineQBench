# agents/factory.py

from eval.answer_type import ANSWER_TYPE
from eval.agent import *

EVALUATE_REGISTRY = {
    ANSWER_TYPE.REPORT: ReportAgent,
    ANSWER_TYPE.LIST: ListAgent,
    ANSWER_TYPE.STEP: StepAgent,
}


class EvaluateFactory:

    @staticmethod
    def create_agent(agent_type: ANSWER_TYPE):
        agent_class = EVALUATE_REGISTRY.get(agent_type)
        if not agent_class:
            raise ValueError(f"No agent found for query type: {agent_type}")
        return agent_class()
