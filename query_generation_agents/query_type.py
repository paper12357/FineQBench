from enum import Enum

class QUERY_TYPE(Enum):
    RETRIEVE_INFO = "Retrieve specific information from the dataset."
    RETRIEVE_FILTER = "Retrieve information based on specific filters or conditions."
    RETRIEVE_FILTER_SIMPLE = "Only use data in databases."
    LOGIC_MULTIHOP = "Multiple logical reasoning steps to get the answer."
    REPORT_SIMPLE = "Generate a simple report based on the data."
    REPORT_COMPARE = "Generate a comparative report based on multiple data points."
    REPORT_TIME_SERIES = "Generate a time series report based on data trends over time."
    ROBUST_WRONG_QUESTION = "The question is incorrect or invalid."
    LOGIC_SIMPLE = "Single logical reasoning step to get the answer."
    LOGIC_CALCULATION = "Involves calculations to derive the answer."
    ROBUST_AMBIGUITY = "The question is ambiguous and can have multiple interpretations."
    ROBUST_NO_ANSWER = "The answer is not present in the dataset."