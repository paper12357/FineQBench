## FineQBench

This repository contains the code, data, and evaluation pipeline for **FineQBench**, a capability-oriented benchmark for evaluating **data agents**—LLM-driven systems that answer complex analytical queries over heterogeneous data.

### Features

- The benchmark defines **fine-grained query taxonomy** that distinguishes queries by their underlying information needs and reasoning requirements, enabling targeted analysis of agent capabilities.
- Queries are constructed in a **capability-oriented manner**, combining manually designed query templates and automated query-answer pairs generation to ensure that correct answers require appropriate tool use and reasoning. 
- The benchmark covers **diverse domains and data modalities**, including structured, semi-structured, and unstructured sources, to reflect realistic heterogeneous data environments. 
- The overall framework is **modular and extensible**, allowing new data sources, query types, agents, and evaluation metrics to be integrated with minimal effort.

### Environment Setup

#### System Requirements

- Python: 3.10+
- OS: Windows, Linux, macOS

#### Installation

Command for installation:

```
conda create -n fineqbench python=3.10
conda activate fineqbench
pip install -r requirements.txt
```

#### API Configuration

Set up your API keys for LLM and Web search access:

```
# Windows
setx OPENAI_API_KEY "your_openai_api_key"
setx OPENROUTER_API_KEY "your_openrouter_api_key"
setx SERPAPI_KEY "your_serpapi_api_key"

# macOS / Linux
export OPENAI_API_KEY="your_openai_api_key"
export OPENROUTER_API_KEY="your_openrouter_api_key"
export SERPAPI_KEY="your_serpapi_api_key"
```

### Example Usage

We provide an example usage.  You can start using FineQBench with the following command:

```
# Query Generation
python example/generate_query.py

# Evaluation
python example/evaluate.py
```

### Dataset Availability

This repository includes the majority of the data required to use FineQBench. However, some data is not included due to size consideration:

- The geography domain contains a large number of image files, which are not included in this repository. 
- The sentence embedding model specified in the configuration  (`models/all-MiniLM-L6-v2`) is not included in the repository.  This model can be obtained from standard public model repositories.
- Pre-built vector databases are not included.  They can be generated locally using the provided script: `dataset/tools/generate_vector.py`

The complete datasets and resources will be made publicly available after paper acceptance.

### Query Types

In the code, query types are referred to by short abbreviations. The mapping between these abbreviations and the query type names used in the paper is documented in the comments at the top of `example/config.py`.  

### Repository Structure

```
FineQBench
├── example/ # example usage
│ └── generate_query.py # generate QA pairs from query template
│ └── evaluate.py # evaluate data agents
│ └── example_template.jsonl # example query template
│ └── config.py # defines datasets, queries and LLM backends
├── dataset/ # data source
├── tools/ # tool layer and unified interface
├── query_generation_agents/ # query generation agent implementations
├── query/ # QA pairs of FineQBench
├── data_agents/ # data agent implementations
├── eval/ # evaluation scripts
├── requirements.txt # python dependencies
└── README.md
```
