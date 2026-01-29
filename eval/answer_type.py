from enum import Enum

class ANSWER_TYPE(Enum):
    REPORT = "report"
    LIST = "list"
    STEP = "step"
    ROUGE = "rouge"
    SUBTASK_COMP = "subtask"
    TOOL_CALL = "tools"