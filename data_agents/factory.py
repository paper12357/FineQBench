# agents/factory.py

from data_agents.agent_type import AGENT_TYPE
from data_agents.agent import *

AGENT_REGISTRY = {
    AGENT_TYPE.TOOL_USE: ToolUseAgent,
    AGENT_TYPE.PLANNING: PlanningAgent,
}


class AgentFactory:
    @staticmethod
    def create_agent(agent_type: AGENT_TYPE, dataset: dict = None, dataset_dir: str = None,
                     tools: list = None, llm_model: str = None):
        agent_class = AGENT_REGISTRY.get(agent_type)
        if not agent_class:
            raise ValueError(f"No agent found for query type: {agent_type}")
        return agent_class(dataset=dataset, dataset_dir=dataset_dir,
                           tools=tools, llm_model=llm_model)
