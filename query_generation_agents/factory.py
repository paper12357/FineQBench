# agents/factory.py

from query_generation_agents.query_type import QUERY_TYPE
from query_generation_agents.agent import *

AGENT_REGISTRY = {
    QUERY_TYPE.RETRIEVE_INFO: RetrieveInfoAgent,
    QUERY_TYPE.RETRIEVE_FILTER: RetrieveFilterAgent,
    QUERY_TYPE.RETRIEVE_FILTER_SIMPLE: RetrieveFilterSimpleAgent,
    QUERY_TYPE.LOGIC_MULTIHOP: LogicMultihopAgent,
    QUERY_TYPE.REPORT_SIMPLE: ReportSimpleAgent,
    QUERY_TYPE.REPORT_COMPARE: ReportCompareAgent,
    QUERY_TYPE.REPORT_TIME_SERIES: ReportTimeSeriesAgent,
    QUERY_TYPE.ROBUST_WRONG_QUESTION: RobustWrongQuestionAgent,
    QUERY_TYPE.LOGIC_SIMPLE: LogicSimpleAgent,
    QUERY_TYPE.LOGIC_CALCULATION: LogicCalculationAgent,
    QUERY_TYPE.ROBUST_AMBIGUITY: RobustAmbiguityAgent,
    QUERY_TYPE.ROBUST_NO_ANSWER: RobustNoAnswerAgent,
}


class AgentFactory:
    @staticmethod
    def create_agent(query_type: QUERY_TYPE):
        agent_class = AGENT_REGISTRY.get(query_type)
        if not agent_class:
            raise ValueError(f"No agent found for query type: {query_type}")
        return agent_class()
