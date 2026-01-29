from abc import ABC, abstractmethod
from query_generation_agents.query_type import QUERY_TYPE
import os
import json
import ast
import re
from tools.db_manager import DBManager
from tools.openrouter_api import LLMClient
from tools.serp_api import SearchClient
from tools.vec_manager import VectorSearchManager
from tools.file_manager import FileManager
from tools.graph_manager import GraphManager
from tools.lotus_manager import LotusManager
from tools.tools import *

class BaseAgent(ABC):
    def __init__(self, query_type: QUERY_TYPE):
        self.query_type = query_type
        self.tools_desc = {
            "DBExecuteTool":(
                "- Purpose: execute a SQL query against the database.\n"
                "- Args format: {\"query\": \"SELECT ...\"} (SQL must be a single string).\n"
                "- Example: {\"query\": \"SELECT name FROM orders WHERE year=2024\"}\n\n"
                "- Notes: Make sure to understand the database schema from the following info.\n"
                "- Schema info:\n"
            ),        
            "VectorTool":(
                "- Purpose: semantic vector retrieval from internal vector DB (text search).\n"
                "- Args format: {\"query\": \"<search text>\", \"k\": <int>} (k optional defaults to 5).\n"
                "- Example: {\"query\":\"Ronaldo 2014 Champions League\",\"k\":5}\n\n"
                "- Notes: Recommend to use this tool for more relevant infromation. Sometimes it is better than WebTool.\n\n"
            ),
            "FileTool":(
                "- Purpose: search local files and return file paths and previews.\n"
                "- Args format: {\"query\": \"<search text>\", \"k\": <int>} (k optional defaults to 2).\n"
                "- Example: {\"query\":\"Lionel Messi\",\"k\":3}\n\n"
                "- Notes: Useful infomation usually stored in local files, but FileTool return file paths and short previews.\n"
                "         You can get the paths first, and then deal with the full file content in later step using CodeTool.\n\n"
                "- Hints: If the question need calculation, such as 'what is the total...', consider that there may be a relevant csv file, \n"
                "         you need to use FileTool first to find file path and then use CodeTool to do calculation.\n\n"
            ),
            "WebTool":(
                "- Purpose: internet search.\n"
                "- Args format: {\"query\": \"<search text>\", \"k\": <int>}.\n"
                "- Example: {\"query\":\"Has Mbappe won the World Cup\",\"k\":5}\n\n"
            ),          
            "LLMTool":(
                "- Purpose: free-form reasoning, text generation, or to produce intermediate JSON/text.\n"
                "- Args format: {\"query\": \"<instruction or text for LLM>\"}.\n"
                "- Example: {\"query\":\"......Summary above materials in 3~5 sentences.\"}\n\n"
            ),            
            "CodeTool":(
                "- Purpose: execute Python code snippets against a single csv/txt/json file.\n"
                "- Args format: {\"abs_path\": \"<absolute file path>\", \"code\": \"<python code>\"}.\n"
                "- Notes: abs_path must be an absolute path to a single csv, txt or json file.\n"
                "         For code snippets, You must use variable 'data' to access the file content (DataFrame for csv, String for txt, dict for json).\n"
                "         You must store your result in variable 'result'.\n"
                "         eg, you want to get the full content of a txt file, you can write code as:\n"
                "         result = data  # data is the full text content of the file\n"
                "         eg, you want to get the sum of column A in a csv file, you can write code as:\n"
                "         import pandas as pd\n"
                "         result = data['A'].sum()  # data is a pandas DataFrame\n\n"
                "[Important Hint]\nThere are CSV files in this dataset, which may include useful data for questions like 'total', 'how many', 'average', etc.\n"
                "         You may use FileTool first to get the relevant file path, and then use CodeTool to run code to get the answer.\n"

            ),
            "LotusTool":(
                "- Purpose: semantic operators (filter/topk) over Lotus database tables.\n"
                "- Args format: {\"type\": \"filter\"/\"topk\", \"table\": \"<table name>\", \"query\": \"<query text>\", \"k\": <int>}.(k optional only for topk)\n"        
                "- Example: {\"type\":\"filter\",\"table\":\"Player\",\"query\":\"{Player} born in Spain\"}\n\n"
                "- Notes: Each table has several text paragraphs for different entities.\n"
                "         Query text must exactly contain one curly braces with the table name, other part is natural language condition.\n"
                "         The table name in curly braces MUST be the SAME as the 'table' field.\n\n"
                "         For example, if the field 'table' is 'Player Early years', the query must contain '{Player Early years}'.\n\n"
                "- Tables: (all valiable table names in Lotus database)\n"
            ),
            "GraphTool":(
                "- Purpose: search entities or relations in knowledge graph database.\n"
                "- Args format: for entity search: {\"type\":\"entity\",\"entity1\":\"<entity name>\"};\n"
                "               for relation search: {\"type\":\"relation\",\"entity1\":\"<entity1>\",\"entity2\":\"<entity2>\"}.\n"
                "- Example: {\"type\":\"relation\",\"entity1\":\"Lionel Messi\",\"entity2\":\"Barcelona\"}\n\n"
            )
        }

    def get_db_schema(self, db_manager) -> str:
        prompt = ""
        tables = db_manager.list_tables()
        for table in tables:
            prompt += f"Table: {table}\n"
            fields = db_manager.list_fields(table)
            prompt += "Fields:\n"
            for field in fields:
                prompt += f"  - {field[1]} ({field[2]})\n"
            sample_data = db_manager.query_all(table, limit=20)
            prompt += "Sample Data:\n"
            for row in sample_data:
                prompt += f"  - {row}\n"
            prompt += "\n"
        return prompt

    def get_db_all_full_name(self, db_manager) -> str:
        prompt = ""
        tables = db_manager.list_tables()
    
        for table in tables:
            full_names = db_manager.query_by_sql(f"SELECT full_name FROM {table}")
            prompt += f"Table: {table}\n"
            prompt += "Full Names:\n"
            for name in full_names:
                prompt += f"  - {name[0]}\n"
            prompt += "\n"

        return prompt


class RetrieveInfoAgent(BaseAgent):
    def __init__(self):
        super().__init__(QUERY_TYPE.RETRIEVE_INFO)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" in tools:
            db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()
        serp_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset["vec_db_path"]),
            collection_name=dataset["collection_name"],
            model_name=os.path.join(dataset_dir, dataset["embedding_model"])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, p) for p in dataset["file_path"]]
        )
        if "graph" in tools:
            graph_manager = GraphManager(
                db_path=[os.path.join(dataset_dir, fp) for fp in dataset['graph_db_path']]
            )

        prompt = ""

        if "db" in tools:
            prompt += "[Data base schema and all full names]\n"
            prompt += self.get_db_schema(db_manager)
            prompt += self.get_db_all_full_name(db_manager)
        
        prompt += f"[Query Template]\n{template}\n\n"

        prompt += "[Generate QA Pairs]\n"

        prompt += "Please fill {n} questions from the query template based on the above database schema and valid full names. "\
                    "Return 2*{n} rows. The first row is the first question, the second row is the corresponding full name (to fill the placeholder), and so on.\n"\
                    "Do not return blank lines or any unnecessary content. \n".format(n=n)
        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])
        print("[INFO] LLM Response:\n", response)

        qa_pairs = []

        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                print(f"[INFO] Generating answer for pair {i//2 + 1}/{n}")
                question = lines[i].strip()
                full_name = lines[i+1].strip()

                prompt = ""
                prompt += f"[Question]\n{question}\n\n"

                prompt = f"[Internet Search Results for '{question}']\n"
                try:
                    prompt += serp_client.search(query=question, num_results=10)
                except Exception as e:
                    print(f"[WARN] Internet search failed for question '{question}': {e}")
                prompt += "\n\n"

                prompt += f"[Vector Search Results for '{question}']\n"
                prompt += vec_manager.search(query_text=question, top_k=10)
                prompt += "\n\n"

                file_paths = file_manager.fetch_file(key=full_name, limit=1)

                for path in file_paths:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"
                
                prompt += "Please use above information and your knowledge "\
                            "to answer the question: '{question}'. "\
                            "Only return one line which is the answer, do not include any extra content or links.\n".format(question=question)

                while True:
                    try:
                        response = llm_client.call_llm(prompt)
                        break
                    except Exception as e:
                        print(f"[ERROR] Answer generation failed: {e}")

                answer = response.strip()

                qa_pairs.append((question, answer))
                
        return qa_pairs


class RetrieveFilterSimpleAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.RETRIEVE_FILTER_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" not in tools:
            print("[ERROR] RETRIEVE_FILTER_SIMPLE agent requires 'db' tool.")
            return []
        
        db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()

        prompt = "[Data base schema and sample data]\n"
        
        prompt += self.get_db_schema(db_manager)
        
        prompt += f"[Query Template]\n{template}\n\n"

        prompt += "[Generate QA Pairs]\n"

        prompt += "Please fill {n} questions from the query template based on the above database schema and sample data. "\
                    "For each question, provide a concise and accurate sql query that retrieves the answer from the database. "\
                    "Try to ensure that the number of rows that can pass the condition is around 10-20%.\n"\
                    "Return 2*{n} rows. The first row is the first question, the second row is the corresponding SQL statement, and so on.\n"\
                    "Do not return blank lines or any unnecessary content. \n".format(n=n)
        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])

        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected, expected n pairs of question and sql.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                sql = lines[i+1].strip()
                try:
                    answer_rows = db_manager.query_by_sql(sql)
                    answer = []
                    for r in answer_rows:
                        answer.append(r[0])
                    qa_pairs.append((question, answer))
                except Exception as e:
                    print(f"[ERROR] Failed to execute SQL: {sql}, error: {e}")

        return qa_pairs


class RetrieveFilterAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.RETRIEVE_FILTER)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" not in tools:
            print("[ERROR] RETRIEVE_FILTER agent requires 'db' tool.")
            return []
        db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()

        prompt = "[Data base schema and sample data]\n"
        
        prompt += self.get_db_schema(db_manager)
        
        prompt += f"[Query Template]\n{template}\n\n"

        prompt += "[Generate QA Pairs]\n"

        prompt += "Please fill {n} questions from the query template based on the above database schema and sample data. "\
                    "For each question, provide a sql query that retrieves the eligible answer from the database. "\
                    "Note that there may be some conditions in the template that cannot be determined based on the content in the database. "\
                    "Please ignore these conditions for now.\n"\
                    "Try to ensure that the number of rows that can pass the condition is around 10-20%.\n"\
                    "Return 2*{n} rows. The first row is the first question, the second row is the corresponding SQL statement, and so on.\n"\
                    "Do not return blank lines or any unnecessary content. \n".format(n=n)
        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])

        print("[INFO] LLM Response:\n", response)

        prompt = ""

        prompt += f"[Questions and SQL Statements]\n{response}\n\n"

        prompt += "These are my questions and SQL statements.\n"\
                "After filtering out some answers using SQL statements, there may be some additional conditions.\n"\
                "Please summarize the conditions that still need to be met after filtering through SQL statements.\n"\
                "Return only one line of conditions without any extra content.\n"\
                "Example: The player has scored in the European Championship. "\
        
        while True:
            try:
                add_cond = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Additional conditions generation failed: {e}")

        print("[INFO] LLM Additional Conditions Response:\n", add_cond)

        qa_pairs = []
        
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected, expected n pairs of question and sql.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                sql = lines[i+1].strip()
                try:
                    answer_rows = db_manager.query_by_sql(sql)
                    answer = []
                    for r in answer_rows:
                        answer.append(r[0])
                    
                    prompt = ""

                    prompt += f"[Condition]\n{add_cond}\n\n"

                    prompt += f"[Alternative Answer]\n{answer}\n\n"

                    prompt += "Please use online search and your knowledge "\
                                "to filter out items from the currently provided answer list "\
                                "that fully meet the condition.\n"\
                                "Only return one line, which is the answer list, "\
                                "in the original format for me to parse directly using Python.\n"
                    while True:
                        try:
                            refined_response = llm_client.call_llm(prompt)
                            break
                        except Exception as e:
                            print(f"[ERROR] Refinement generation failed: {e}")
                    
                    #print(f"[INFO] Refinement LLM Response:\n {refined_response}")
                    
                    refined_answer = ast.literal_eval(refined_response.strip())
                
                    qa_pairs.append((question, refined_answer))

                    print(f"[INFO] Original answer: {answer}, Refined answer: {refined_answer}")
                    with open("eval_logs/refinement_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"Question: {question}\n")
                        f.write(f"SQL: {sql}\n")
                        f.write(f"Original answer: {answer}\n")
                        f.write(f"Refined answer: {refined_answer}\n")
                        f.write("\n")

                except Exception as e:
                    print(f"[ERROR] Failed to generate answer: {sql}, error: {e}")

        return qa_pairs
    
class LogicMultihopAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.LOGIC_MULTIHOP)

    def generate(self, template: dict, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        self.registry = ToolRegistry()

        if "db" in tools:
            db_manager = DBManager(
                db_path=os.path.join(dataset_dir, dataset['db_path'])
            )
            self.registry.register_tool(DBExecuteTool(db_manager))
            self.registry.register_tool(DBSchemaTool(db_manager))
            self.tools_desc["DBExecuteTool"] += self.get_db_schema(db_manager)
        else:
            self.tools_desc.pop("DBExecuteTool", None)
        if "lotus" in tools:
            lotus_manager = LotusManager(
                db_path=os.path.join(dataset_dir, dataset['lotus_db_path'])
            )
            self.registry.register_tool(LotusTool(lotus_manager))
            self.tools_desc["LotusTool"] += lotus_manager.get_table_names()
        else:
            self.tools_desc.pop("LotusTool", None)
        if "graph" in tools:
            graph_manager = GraphManager(
                db_path=[os.path.join(dataset_dir, fp) for fp in dataset['graph_db_path']]
            )
            self.registry.register_tool(GraphTool(graph_manager))
        else:
            self.tools_desc.pop("GraphTool", None)

        llm_client = LLMClient()
        search_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset['vec_db_path']),
            collection_name=dataset['collection_name'],
            model_name=os.path.join(dataset_dir, dataset['embedding_model'])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, fp) for fp in dataset['file_path']]
        )
        self.registry.register_tool(VectorTool(vec_manager))
        self.registry.register_tool(FileTool(file_manager))
        self.registry.register_tool(WebTool(search_client))
        self.registry.register_tool(LLMTool(llm_client))


        template_txt = template.get("template", "")
        description = template.get("description", "")
        essential_tool_calls = template.get("essential_tool_calls", [])

        decompose_prompt = (
            "You are a reasoning assistant."
            "Your task is to decompose the given query template into 2~5 logical steps necessary to answer it.\n"
            "Use [placeholders] for entities or attributes in each step.\n"
            "Return each step as a single line. Do not include extra commentary or blank lines.\n\n"
            "Example template:\n"
            "Where is the home stadium of the last club [Player] played for?\n"
            "Example output:\n"
            "What is the last club [Player] play for?\n"
            "Where is the home stadium of [Club]?\n"

            f"Query Template:{template_txt}\n Description:{description}\n\n"
            "The description provides additional guidance to the reasoning assistant, "
            "serving as a reference for how to plan the reasoning process and split the query into clear, sequential steps."
        )
        while True:
            try:
                decomposition = llm_client.call_llm(decompose_prompt, model="gpt-4o")
                break
            except Exception as e:
                print(f"[ERROR] Decomposition failed: {e}")

        decomposition = "\n".join([line for line in decomposition.split("\n") if line.strip() != ""])

        print("[INFO] LLM Decomposition Response:\n", decomposition)

        steps = [line.strip() for line in decomposition.strip().split("\n") if line.strip()]

        prompt = ""
        if "db" in tools:
            prompt += "[Data base schema and all full names]\n"
        
            prompt += self.get_db_schema(db_manager)

            prompt += self.get_db_all_full_name(db_manager)
        
        prompt += f"[Query Template]\n{steps[0]}\n\n"

        prompt += "[Generate QA Pairs]\n"

        prompt += "Please fill {n} questions from the query template based on the above database schema and valid full names. "\
                    "Return 2*{n} rows. The first row is the first question, the second row is the entity you use to fill the placeholder, and so on.\n"\
                    "Do not return blank lines or any unnecessary content. \n".format(n=n)

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])

        print("[INFO] Filled Questions:\n", response)

        qa_pairs = []

        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                print(f"[INFO] Generating multihop answer for pair {i//2 + 1}/{n}")
                try:
                    full_name = lines[i+1].strip()

                    questions = []

                    questions.append(re.sub(r"\[[^\]]+\]", full_name, template_txt))

                    questions.append(lines[i].strip())
                    for step in steps[1:]:
                        questions.append(step)

                    answers = []       

                    prev_log = []

                    for q in questions[1:]:
                        prompt = (
                            "You are a reasoning assistant. Using the previous steps' Q&A, fill in the placeholders in the current step question template."
                            "Only return the completed question. Do not include extra content.\n\n"
                            "Example:\nPrevious Steps:\n"
                            "Q: What is the last club Messi played for?\n"
                            "A: Barcelona\n"
                            "Current Step Template:\n"
                            "Where is the home stadium of [Club]?\n"
                            "Completed Question:\n"
                            "Where is the home stadium of Barcelona?\n\n"
                            f"Previous Steps:\n{prev_log}\nCurrent Step Template:\n{q}"
                        )

                        while True:
                            try:
                                question = llm_client.call_llm(prompt).strip()
                                break
                            except Exception as e:
                                print(f"[ERROR] Fill current step question failed: {e}")

                        prompt = ""
                        prompt += f"[Question]\n{question}\n\n"

                        prompt = f"[Internet Search Results for '{question}']\n"
                        prompt += search_client.search(query=question, num_results=10)
                        prompt += "\n\n"

                        prompt += f"[Vector Search Results for '{question}']\n"
                        prompt += vec_manager.search(query_text=question, top_k=10)
                        prompt += "\n\n"

                        file_paths = file_manager.fetch_file(key=full_name, limit=2)

                        for path in file_paths:
                            with open(path, "r", encoding="utf-8") as f:
                                content = f.read()
                                prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"

                        tool_info = "\n".join(
                            [f"- {name}:\n{desc}" for name, desc in self.tools_desc.items()]
                        )
                        essential_tool_call_prompt = (
                            "You are a multi-tool reasoning agent. "
                            "For the current logical step (which is a sub-step of the original query), select exactly ONE tool to execute and provide its arguments in a JSON object.\n"
                            "Only output a single parseable JSON object; do not include code blocks or extra content.\n\n"
                            f"Original Query:\n{template_txt}\n\n"
                            f"Available Tools and Usage:\n{tool_info}\n\n"
                            f"Reference Essential Tool Calls:\n{essential_tool_calls}\n\n"
                            "Instructions:\n"
                            "- Choose the most appropriate operator for the current logical step.\n"
                            "- Use previously executed steps and results to make an informed decision.\n"
                            "- Consider the Original Query Template and Essential Tool Calls as references, but selection should be based on the current step.\n"
                            "- Return JSON exactly in this shape: {\"operator\": \"<Toolname>\", \"args\": { ... }}, where <Toolname> must be one of the available tools.\n"
                            f"Previously executed steps and results:\n{prev_log}\n\n"
                            f"Current Logical Step:\n{question}\n"
                        )

                        while True:
                            try:
                                tool_call_response = llm_client.call_llm(essential_tool_call_prompt, model="gpt-4o")
                                tool_call_json = json.loads(tool_call_response.strip())
                                break
                            except Exception as e:
                                print(f"[ERROR] Tool selection failed: {e}")

                        try:
                            tool_name = tool_call_json["operator"]
                            input_payload = tool_call_json["args"]
                            tool_result = self.registry.call_tool(tool_name, input_payload)
                        except Exception as e:
                            print(f"[ERROR] Tool execution failed: {e}")
                            tool_result = "Tool execution failed."
                        
                        prompt += f"[Tool ({tool_name}) Result]\n{tool_result}\n\n"
                        

                        prompt += f"The original question is: '{questions[0]}'\n\n"
                        prompt += "[Previous Steps]\n"
                        for log in prev_log:
                            prompt += f"Q: {log['question']}\nA: {log['answer']}\n"
                        
                        prompt += "Now, Please use above information and your knowledge "\
                                    "to answer the CURRENT STEP question: '{question}'. "\
                                    "Only return one line which is the answer(eg: a name, or a number), do not include any extra content or links.\n".format(question=question)
                        
                        while True:
                            try:
                                response = llm_client.call_llm(prompt)
                                break
                            except Exception as e:
                                print(f"[ERROR] Answer generation failed: {e}")
                        
                        a = response.strip()

                        answers.append(a)
                        prev_log.append({
                            "question": question,
                            "answer": a
                        })

                        full_name = a  

                        print(f"[INFO] Question: {question}, Answer: {a}")


                    qa_pairs.append((questions, answers))
                except Exception as e:
                    print(f"[ERROR] Failed to generate multihop QA pair, error: {e}")

                
        return qa_pairs
    


class ReportSimpleAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" in tools:
            db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()
        serp_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset["vec_db_path"]),
            collection_name=dataset["collection_name"],
            model_name=os.path.join(dataset_dir, dataset["embedding_model"])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, p) for p in dataset["file_path"]]
        )
        prompt = ""
        if "db" in tools:
            prompt = "[Database Schema and all full names]\n"
            prompt += self.get_db_schema(db_manager)
            prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"
        prompt += "[Generate QA Pairs]\n"
        prompt += (
            f"Please fill {n} questions from the query template based on the above database information. "
            f"For each filled question, output two lines:\n"
            f"  1. The completed question with all placeholders replaced.\n"
            f"  2. All names that were used to fill the placeholders (comma-separated, in the same order as they appear in the template).\n"
            f"Return exactly 2*{n} lines in total (question + names alternating). "
            "Do not include explanations, blank lines, or any extra text.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])
        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                full_names = lines[i + 1].strip().split(",")

                prompt = f"[Question]\n{question}\n\n"

                prompt += f"[Internet Search Results for '{question}']\n"
                prompt += serp_client.search(query=question, num_results=10)
                prompt += "\n\n"

                prompt += f"[Vector Search Results for '{question}']\n"
                prompt += vec_manager.search(query_text=question, top_k=10)
                prompt += "\n\n"

                for full_name in full_names:
                    file_paths = file_manager.fetch_file(key=full_name)
                    for path in file_paths:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"

                prompt += (
                    "Please synthesize the above materials and your knowledge to write a concise, factual report "
                    f"answering the question: '{question}'. "
                    "Write in one short paragraph, no lists or links.\n"
                )

                while True:
                    try:
                        response = llm_client.call_llm(prompt)
                        break
                    except Exception as e:
                        print(f"[ERROR] Report generation failed: {e}")

                qa_pairs.append((question, response.strip()))

        return qa_pairs
    

class ReportCompareAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" in tools:
            db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()
        serp_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset["vec_db_path"]),
            collection_name=dataset["collection_name"],
            model_name=os.path.join(dataset_dir, dataset["embedding_model"])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, p) for p in dataset["file_path"]]
        )
        prompt = ""
        if "db" in tools:
            prompt = "[Database Schema and all full names]\n"
            prompt += self.get_db_schema(db_manager)
            prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"
        prompt += "[Generate QA Pairs]\n"
        prompt += (
            f"Please fill {n} questions from the query template based on the above database information. "
            f"Note that the key of the question is comparison. Please use your knowledge to select appropriate subjects for comparison, rather than choosing them arbitrarily.\n"
            f"For each filled question, output two lines:\n"
            f"  1. The completed question with all placeholders replaced.\n"
            f"  2. All names that were used to fill the placeholders (comma-separated, in the same order as they appear in the template).\n"
            f"Return exactly 2*{n} lines in total (question + names alternating). "
            "Do not include explanations, blank lines, or any extra text.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])
        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                full_names = lines[i + 1].strip().split(",")

                prompt = f"[Question]\n{question}\n\n"

                prompt += f"[Internet Search Results for '{question}']\n"
                prompt += serp_client.search(query=question, num_results=10)
                prompt += "\n\n"

                prompt += f"[Vector Search Results for '{question}']\n"
                prompt += vec_manager.search(query_text=question, top_k=10)
                prompt += "\n\n"

                for full_name in full_names:
                    file_paths = file_manager.fetch_file(key=full_name)
                    for path in file_paths:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"

                prompt += (
                    "Please synthesize the above materials and your knowledge to write a concise, factual report "
                    f"answering the question: '{question}'. "
                    "Write in one short paragraph, no lists or links.\n"
                )

                while True:
                    try:
                        response = llm_client.call_llm(prompt)
                        break
                    except Exception as e:
                        print(f"[ERROR] Report generation failed: {e}")

                qa_pairs.append((question, response.strip()))

        return qa_pairs
    


class ReportTimeSeriesAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" in tools:
            db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()
        serp_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset["vec_db_path"]),
            collection_name=dataset["collection_name"],
            model_name=os.path.join(dataset_dir, dataset["embedding_model"])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, p) for p in dataset["file_path"]]
        )
        prompt = ""
        if "db" in tools:
            prompt = "[Database Schema and all full names]\n"
            prompt += self.get_db_schema(db_manager)
            prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"
        prompt += "[Generate QA Pairs]\n"
        prompt += (
            f"Please fill {n} questions from the query template based on the above database information. "
            f"Note that the key to these questions is time-based analysis, so please try to select representative names when filling in the questions.\n"
            f"For each filled question, output two lines:\n"
            f"  1. The completed question with all placeholders replaced.\n"
            f"  2. All names that were used to fill the placeholders (comma-separated, in the same order as they appear in the template).\n"
            f"Return exactly 2*{n} lines in total (question + names alternating). "
            "Do not include explanations, blank lines, or any extra text.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])
        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                full_names = lines[i + 1].strip().split(",")

                prompt = f"[Question]\n{question}\n\n"

                prompt += f"[Internet Search Results for '{question}']\n"
                prompt += serp_client.search(query=question, num_results=10)
                prompt += "\n\n"

                prompt += f"[Vector Search Results for '{question}']\n"
                prompt += vec_manager.search(query_text=question, top_k=10)
                prompt += "\n\n"

                for full_name in full_names:
                    file_paths = file_manager.fetch_file(key=full_name)
                    for path in file_paths:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"

                prompt += (
                    "Please synthesize the above materials and your knowledge to write a concise, factual report "
                    f"answering the question: '{question}'. "
                    "Write in one short paragraph, no lists or links.\n"
                )

                #with open("debug_prompt.txt", "w", encoding="utf-8") as f:
                #    f.write(prompt)
                while True:
                    try:
                        response = llm_client.call_llm(prompt)
                        break
                    except Exception as e:
                        print(f"[ERROR] Report generation failed: {e}")

                qa_pairs.append((question, response.strip()))

        return qa_pairs
    
class RobustWrongQuestionAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" not in tools:
            print("[ERROR] ROBUST_WRONG_QUESTION agent requires 'db' tool.")
            return []
        db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()

        prompt = "[Data base schema and all full names]\n"
        prompt += self.get_db_schema(db_manager)
        prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"

        prompt += "[Generate Query]\n"

        prompt += (
            f"Please fill {n} questions from the query template, but intentionally use incorrect or illogical names for the placeholders. "
            "Only replace the contents inside the square brackets with entities in the database(if possible), "
            "such as players with the wrong era, clubs that never existed, or mismatched combinations. "
            "Do NOT modify or paraphrase any other part of the template text. Keep punctuation, structure, and wording identical to the original template.\n"
            f"Return exactly {n} lines, each containing one completed (but incorrect) question. "
            "Do not include explanations, reasoning, blank lines, or any extra content.\n"
        )   

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])

        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        
        lines = response.strip().split("\n")
        if len(lines) != n:
            print("[WARNING] LLM response format unexpected, expected n pairs of question and sql.")
        else:
            for i in range(0, len(lines), 1):
                question = lines[i].strip()
                answer = "The prerequisite of the question is wrong, cannot be answered."
                qa_pairs.append((question, answer))

        return qa_pairs
    

class LogicSimpleAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" in tools:
            db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()
        serp_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset["vec_db_path"]),
            collection_name=dataset["collection_name"],
            model_name=os.path.join(dataset_dir, dataset["embedding_model"])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, p) for p in dataset["file_path"]]
        )
        prompt = ""
        if "db" in tools:
            prompt += "[Database Schema and all full names]\n"
            prompt += self.get_db_schema(db_manager)
            prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"
        prompt += "[Generate QA Pairs]\n"
        prompt += (
            f"Please fill {n} questions from the query template based on the above database information. "
            f"For each filled question, output two lines:\n"
            f"  1. The completed question with all placeholders replaced.\n"
            f"  2. All names that were used to fill the placeholders (comma-separated, in the same order as they appear in the template).\n"
            f"Return exactly 2*{n} lines in total (question + names alternating). "
            "Do not include explanations, blank lines, or any extra text.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")
        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])
        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                full_names = lines[i + 1].strip().split(",")

                prompt = f"[Question]\n{question}\n\n"

                prompt += f"[Internet Search Results for '{question}']\n"
                prompt += serp_client.search(query=question, num_results=10)
                prompt += "\n\n"

                prompt += f"[Vector Search Results for '{question}']\n"
                prompt += vec_manager.search(query_text=question, top_k=10)
                prompt += "\n\n"

                for full_name in full_names:
                    file_paths = file_manager.fetch_file(key=full_name)
                    for path in file_paths:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"

                prompt += (
                    f"Please synthesize the above materials and your knowledge to answer: '{question}'\n"
                    "Write in one line, no lists or links.\n"
                )

                #with open("debug_prompt.txt", "w", encoding="utf-8") as f:
                #    f.write(prompt)

                while True:
                    try:
                        response = llm_client.call_llm(prompt)
                        break
                    except Exception as e:
                        print(f"[ERROR] Answer generation failed: {e}")
                qa_pairs.append((question, response.strip()))

        return qa_pairs
    

class LogicCalculationAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" in tools:
            db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()
        serp_client = SearchClient()
        vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset["vec_db_path"]),
            collection_name=dataset["collection_name"],
            model_name=os.path.join(dataset_dir, dataset["embedding_model"])
        )
        file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, p) for p in dataset["file_path"]]
        )
        if "db" in tools:
            prompt = "[Database Schema and all full names]\n"
            prompt += self.get_db_schema(db_manager)
            prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"
        prompt += "[Generate QA Pairs]\n"
        prompt += (
            f"Please fill {n} questions from the query template based on the above database information. "
            f"For each filled question, output two lines:\n"
            f"  1. The completed question with all placeholders replaced.\n"
            f"  2. All names that were used to fill the placeholders (comma-separated, in the same order as they appear in the template).\n"
            f"Return exactly 2*{n} lines in total (question + names alternating). "
            "Do not include explanations, blank lines, or any extra text.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])
        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        lines = response.strip().split("\n")
        if len(lines) != 2 * n:
            print("[WARNING] LLM response format unexpected.")
        else:
            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                full_names = lines[i + 1].strip().split(",")

                prompt = f"[Question]\n{question}\n\n"

                prompt += f"[Internet Search Results for '{question}']\n"
                prompt += serp_client.search(query=question, num_results=10)
                prompt += "\n\n"

                prompt += f"[Vector Search Results for '{question}']\n"
                prompt += vec_manager.search(query_text=question, top_k=10)
                prompt += "\n\n"

                for full_name in full_names:
                    file_paths = file_manager.fetch_file(key=full_name, limit=2)
                    for path in file_paths:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                            prompt += f"[File Content from {os.path.basename(path)}]\n{content}\n\n"

                prompt += (
                    f"Please synthesize the above materials and your knowledge to answer: '{question}'\n"
                    "Write in one line, no lists or links.\n"
                )

                while True:
                    try:
                        response = llm_client.call_llm(prompt)
                        break
                    except Exception as e:
                        print(f"[ERROR] Answer generation failed: {e}")

                qa_pairs.append((question, response.strip()))

        return qa_pairs
    


class RobustAmbiguityAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" not in tools:
            print("[ERROR] ROBUST_AMBIGUITY agent requires 'db' tool.")
            return []
        db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()

        prompt = "[Data base schema and all full names]\n"
        prompt += self.get_db_schema(db_manager)
        prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"

        prompt += "[Generate Query]\n"

        prompt += (
            f"Please fill {n} ambiguous questions from the query template by intentionally introducing entity ambiguity or unclear references. "
            "Only modify the placeholders inside square brackets to create natural but ambiguous cases — for example, names shared by multiple people "
            "(e.g., 'Ronaldo'), or vague temporal or contextual expressions. "
            "Do NOT correct or clarify the ambiguity. Keep punctuation, structure, and wording identical to the original template.\n"
            f"Return exactly {n} lines, each containing one completed (but ambiguous) question. "
            "Do not include explanations, reasoning, blank lines, or any extra content.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])

        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        
        lines = response.strip().split("\n")
        if len(lines) != n:
            print("[WARNING] LLM response format unexpected, expected n pairs of question and sql.")
        else:
            for i in range(0, len(lines), 1):
                question = lines[i].strip()
                answer = "The question is ambiguous, cannot be answered."
                qa_pairs.append((question, answer))

        return qa_pairs
    

class RobustNoAnswerAgent(BaseAgent):

    def __init__(self):
        super().__init__(QUERY_TYPE.REPORT_SIMPLE)

    def generate(self, template: str, dataset: dict, dataset_dir: str, n: int = 5, tools: list = []):
        if "db" not in tools:
            print("[ERROR] ROBUST_NO_ANSWER agent requires 'db' tool.")
            return []
        db_manager = DBManager(db_path=os.path.join(dataset_dir, dataset["db_path"]))
        llm_client = LLMClient()

        prompt = "[Data base schema and all full names]\n"
        prompt += self.get_db_schema(db_manager)
        prompt += self.get_db_all_full_name(db_manager)

        prompt += f"[Query Template]\n{template}\n\n"

        prompt += "[Generate Query]\n"

        prompt += (
            f"Please fill {n} questions from the query template. "
            "Only fill the placeholders, do not modify any other part of the template. "
            "Ensure that the completed questions CANNOT be answered based on the database content."
            "You should using names that NOT exist in the database to fill the question. "
            f"Return exactly {n} lines, each containing one completed question. "
            "Do not include explanations, reasoning, blank lines, or any extra content.\n"
        )

        while True:
            try:
                response = llm_client.call_llm(prompt)
                break
            except Exception as e:
                print(f"[ERROR] Fill template failed: {e}")

        response = "\n".join([line for line in response.split("\n") if line.strip() != ""])

        print("[INFO] LLM Response:\n", response)

        qa_pairs = []
        
        lines = response.strip().split("\n")
        if len(lines) != n:
            print("[WARNING] LLM response format unexpected, expected n pairs of question and sql.")
        else:
            for i in range(0, len(lines), 1):
                question = lines[i].strip()
                answer = "I can’t find a proper answer for this question."
                qa_pairs.append((question, answer))

        return qa_pairs