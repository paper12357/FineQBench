# Query type abbreviations used in USE_QUERY
# ------------------------------------------------------------
# Information Acquisition
# RTI  : RETRIEVE_INFO            (paper: "Information Extraction")
# RPS  : REPORT_SIMPLE            (paper: "Perspective Summary")
# RPC  : REPORT_COMPARE           (paper: "Comparative Summary")
# RPT  : REPORT_TIME_SERIES       (paper: "Temporal Summary")

# Logical Reasoning
# RTFS : RETRIEVE_FILTER_SIMPLE   (paper: "Simple Filtering")
# RTF  : RETRIEVE_FILTER          (paper: "Complex Filtering")
# LGS  : LOGIC_SIMPLE             (paper: "Logical Reasoning")
# LGC  : LOGIC_CALCULATION        (paper: "Numerical Computation")
# LGM  : LOGIC_MULTIHOP           (paper: "Multi-hop Logic")

# Robustness
# RBW  : ROBUST_WRONG_QUESTION    (paper: "Incorrect Premise")
# RBA  : ROBUST_AMBIGUITY         (paper: "Ambiguous Query")
# RBN  : ROBUST_NO_ANSWER         (paper: "Confidence Evaluation")
# ------------------------------------------------------------

USE_DATASET = {
    "football":{
        "dataset": {
            "db_path": "db/football.db",
            "vec_db_path": "dbvec",
            "collection_name": "football",
            "embedding_model": "models/all-MiniLM-L6-v2",
            "file_path": ["football_clubs",
                          "football_players"],
            "lotus_db_path": "dblotus/lotus_football.json",
            "graph_db_path": ["dbgraph/football/kg_nodes.json", "dbgraph/football/kg_edges.json"],
        },
        "tools": ["db", "lotus", "graph"],
        "dataset_dir": "dataset",
    },
    "film":{
        "dataset": {
            "db_path": "db/film.db",
            "vec_db_path": "dbvec",
            "collection_name": "film",
            "embedding_model": "models/all-MiniLM-L6-v2",
            "lotus_db_path": "dblotus/lotus_film.json",
            "file_path": ["film",
                          "filmer"]
        },
        "tools": ["db", "lotus"],
        "dataset_dir": "dataset",
    },
    "geography":{
        "dataset": {
            "db_path": "db/geography.db",
            "vec_db_path": "dbvec",
            "collection_name": "geography",
            "embedding_model": "models/all-MiniLM-L6-v2",
            "file_path": ["geography"]
        },
        "tools": ["db"],
        "dataset_dir": "dataset",
    },
}

USE_QUERY = {
    "football":{
        "RTI": "query/football/RTI.jsonl",
        "RTFS": "query/football/RTFS.jsonl",
        "RTF": "query/football/RTF.jsonl",
        "RPS": "query/football/RPS.jsonl",
        "RPC": "query/football/RPC.jsonl",
        "RPT": "query/football/RPT.jsonl",
        "LGS": "query/football/LGS.jsonl",
        "LGC": "query/football/LGC.jsonl",
        "LGM": "query/football/LGM.jsonl",
        "RBW": "query/football/RBW.jsonl",
        "RBA": "query/football/RBA.jsonl",
        "RBN": "query/football/RBN.jsonl",
    },
    "film":{
        "RTI": "query/film/RTI.jsonl",
        "RTFS": "query/film/RTFS.jsonl",
        "RTF": "query/film/RTF.jsonl",
        "RPS": "query/film/RPS.jsonl",
        "RPC": "query/film/RPC.jsonl",
        "RPT": "query/film/RPT.jsonl",
        "LGS": "query/film/LGS.jsonl",
        "LGC": "query/film/LGC.jsonl",
        "LGM": "query/film/LGM.jsonl",
        "RBW": "query/film/RBW.jsonl",
        "RBA": "query/film/RBA.jsonl",
        "RBN": "query/film/RBN.jsonl",
    },
    "geography":{
        "RTI": "query/geography/RTI.jsonl",
        "RTFS": "query/geography/RTFS.jsonl",
        "RTF": "query/geography/RTF.jsonl",
        "RPS": "query/geography/RPS.jsonl",
        "RPC": "query/geography/RPC.jsonl",
        "RPT": "query/geography/RPT.jsonl",
        "LGS": "query/geography/LGS.jsonl",
        "LGC": "query/geography/LGC.jsonl",
        "LGM": "query/geography/LGM.jsonl",
        "RBW": "query/geography/RBW.jsonl",
        "RBA": "query/geography/RBA.jsonl",
        "RBN": "query/geography/RBN.jsonl",
    },
}

USE_LLM = {
    "ds": "deepseek/deepseek-chat",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5": "openai/gpt-5",
    "llama-4":"meta-llama/llama-4-maverick",
    "llama-3":"meta-llama/llama-3.1-8b-instruct",
    "claude":"anthropic/claude-sonnet-4",
    "gemini":"google/gemini-2.5-flash",
    "qwen":"qwen/qwen3-vl-235b-a22b-instruct",
}
