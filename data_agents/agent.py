from abc import ABC
from data_agents.agent_type import AGENT_TYPE
import os
import json
import copy
import re
import time
import traceback
from tools.db_manager import DBManager
from tools.openrouter_api import LLMClient
from tools.serp_api import SearchClient
from tools.vec_manager import VectorSearchManager
from tools.file_manager import FileManager
from tools.code_manager import CodeManager
from tools.lotus_manager import LotusManager
from tools.graph_manager import GraphManager
from tools.tools import *

class BaseAgent(ABC):

    def __init__(self, agent_type: AGENT_TYPE):
        self.agent_type = agent_type

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
            sample_data = db_manager.query_all(table, limit=3)
            prompt += "Sample Data:\n"
            for row in sample_data:
                prompt += f"  - {row}\n"
            prompt += "\n"
        return prompt
    
    def detect_entities(self, text: str, llm_client) -> list:
        prompt = (
            "Extract all named entities (people, organizations, etc.) from the following text. "
            "Return the entities in a single line. If more than one, separate them with commas, in order of importance. "
            "No additional explanation or quotes.\n\n"
            f"Text:\n{text}\n\n"
            "Entities:"
        )
        while True:
            try:
                response = llm_client.call_llm(prompt, model=self.llm_model)
                entities = []
                entities = [ent.strip() for ent in response.split(",") if ent.strip()]
                break
            except Exception as e:
                print(f"[ERROR] Failed to parse entities from LLM response: {e}")
        return entities[:3] 

    
class ToolUseAgent(BaseAgent):

    def __init__(self, dataset: dict, dataset_dir: str, tools: list = [], llm_model: str = "deepseek/deepseek-chat"):
        super().__init__(AGENT_TYPE.TOOL_USE)
        
        self.tools = tools
        self.llm_model = llm_model

        self.registry = ToolRegistry()

        self.llm_client = LLMClient()
        self.search_client = SearchClient()
        if "db" in tools:
            self.db_manager = DBManager(
                db_path=os.path.join(dataset_dir, dataset['db_path'])
            )
        self.vec_manager = VectorSearchManager(
            db_path=os.path.join(dataset_dir, dataset['vec_db_path']),
            collection_name=dataset['collection_name'],
            model_name=os.path.join(dataset_dir, dataset['embedding_model'])
        )
        self.file_manager = FileManager(
            file_path=[os.path.join(dataset_dir, fp) for fp in dataset['file_path']]
        )
        if "code" in tools:
            self.code_manager = CodeManager()
        if "lotus" in tools:
            self.lotus_manager = LotusManager(
                db_path=os.path.join(dataset_dir, dataset['lotus_db_path'])
            )
        if "graph" in tools:
            self.graph_manager = GraphManager(
                db_path=[os.path.join(dataset_dir, fp) for fp in dataset['graph_db_path']]
            )

        self.registry.register_tool(LLMTool(self.llm_client, model=self.llm_model))

    def handle_query(self, question: str) -> dict:

        entities = self.detect_entities(question, self.llm_client)

        llm_tool = self.registry.get_tool("LLMTool")

        prompt = f"[User Question]\n{question}\n\n"

        if "db" in self.tools:
            try:
                db_schema = self.get_db_schema(self.db_manager)
                prompt += "[Database Schema and Sample Data]\n" + db_schema + "\n"

                sql_prompt = (
                    "[Database schema]\n"
                    f"{db_schema}\n\n"
                    "Given the above database schema, write an SQL query "
                    f"that can help answer the question: '{question}'. "
                    "Only return the SQL statement."
                )
                sql = llm_tool.run({"query": sql_prompt})["output"].strip()
                prompt += f"[Generated SQL]\n{sql}\n"

                try:
                    db_result = self.db_manager.query_by_sql(sql)
                    if db_result:
                        prompt += "[Database Query Results]\n"
                        for row in db_result[:]:
                            prompt += f"  - {row}\n"
                        prompt += "\n"
                    else:
                        prompt += "[Database Query Results]\n  - (No result)\n\n"
                except Exception as e:
                    prompt += f"[Database Query Error]\n  - {e}\n\n"
            except Exception as e:
                prompt += f"[Database Section Error]\n  - {e}\n\n"

        try:
            vec_results = self.vec_manager.search(query_text=question, top_k=5)
            prompt += f"[Vector Search Results]\n{vec_results}\n\n"
        except Exception as e:
            prompt += f"[Vector Search Error]\n  - {e}\n\n"

        if "lotus" in self.tools:
            try:
                lotus_tables = self.lotus_manager.list_tables()
                lotus_prompt = (
                    "There are some tables in the Lotus database:\n"
                    f"{lotus_tables}\n\n"
                    "Each table has several text paragraphs for different entities.\n"
                    "Given the user question, you need to write a Lotus query to find relevant information.\n\n"
                    "[Example query]: {Player Early years} born in Spain\n"
                    "'Player Early years' is the table name. Lotus query must exactly contain one table name in curly braces, other part is natural language condition.\n\n"
                    f"[User question]: {question}\n"
                    "return only one line Lotus query without any explanation:"
                )
                lotus_query = llm_tool.run({"query": lotus_prompt})["output"].strip()
                prompt += f"[Generated Lotus Query]\n{lotus_query}\n"
                table = re.search(r"\{(.+?)\}", lotus_query).group(1)
                lotus_results = self.lotus_manager.topk(table_name=table, query_text=lotus_query, top_k=3)
                prompt += f"[Lotus Results]\n{lotus_results}\n\n"
            except Exception as e:
                prompt += f"[Lotus Error]\n  - {e}\n\n"

        if "graph" in self.tools:
            try:
                prompt += f"[Detected Entities]\n{entities}\n\n"
                if len(entities) == 1:
                    graph_results = self.graph_manager.search_entity(entities[0])
                if len(entities) > 1:
                    graph_results = self.graph_manager.search_relation(entities[0], entities[1])
                else:
                    graph_results = self.graph_manager.search_entity(question)
                prompt += f"[Graph Results]\n{graph_results}\n\n"
            except Exception as e:
                prompt += f"[Graph Error]\n  - {e}\n\n"

        file_paths = []
        try:
            file_paths = []
            if len(entities) > 0:
                for ent in entities:
                    paths = self.file_manager.fetch_file(key=ent, limit=1)
                    file_paths.extend(paths)
            if len(file_paths) == 0:
                file_paths = self.file_manager.fetch_file(key=question)

            if file_paths:
                for path in file_paths:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    prompt += f"[File Content: {os.path.basename(path)}]\n"
                    prompt += content[:10000] + "\n\n"
            else:
                prompt += "[File Search]\n  - No related files found.\n\n"
        except Exception as e:
            prompt += f"[File Search Error]\n  - {e}\n\n"

        try:
            search_results = self.search_client.search(query=question, num_results=5)
            prompt += f"[Internet Search Results]\n{search_results}\n\n"
        except Exception as e:
            prompt += f"[Internet Search Error]\n  - {e}\n\n"

        if "code" in self.tools:
            try:
                data_files = [p for p in file_paths if os.path.splitext(p)[1].lower() in (".json", ".csv")]

                for abs_path in data_files:
                    try:
                        basename = os.path.basename(abs_path)

                        with open(abs_path, "r", encoding="utf-8") as f:
                            file_text = f.read()

                        llm_prompt = (
                            "You are a careful Python assistant. Below are:\n\n"
                            f"File name: {basename}\n\n"
                            "File content (exact):\n"
                            "'''\n"
                            f"{file_text}\n"
                            "'''\n\n"
                            "User question:\n"
                            f"{question}\n\n"
                            "Task:\n"
                            "Decide whether this file is needed to answer the user question.\n"
                            " - If the file is NOT needed, output exactly the single word: No\n"
                            " - If the file IS needed, output ONLY valid Python code (no explanation, no markdown, no extra text) "
                            "that defines a variable named result which contains the value you want extracted from this file. "
                            "The execution environment will provide a variable named `data` already bound to the file's content "
                            "(for .json it's a Python object from json.load, for .csv it's a pandas.DataFrame). "
                            "You may import standard libraries (e.g. pandas, math, numpy) if needed. "
                            "Do NOT attempt to read the file from disk (the file is already provided in `data`). "
                            "Make sure the code does not rely on any undefined symbols and sets `result` at the end.\n\n"
                            "Remember: output must be exactly either `No` OR a Python program that defines `result` (raw string only! no surrounding text or code block).\n"
                        )

                        #print("[LLM PROMPT]", llm_prompt)

                        while True:
                            try:
                                llm_reply = llm_tool.run({"query": llm_prompt})["output"]
                                break
                            except Exception as e_retry:
                                print(f"[ERROR] LLM call for code part failed, retrying: {e_retry}")

                        print("[LLM REPLY]", llm_reply)

                        if llm_reply.strip() == "No":
                            prompt += f"[Code Execution: {basename}]\nDecision: No (file not needed)\n\n"
                            continue
                        code = llm_reply

                        prompt += f"[Code To Run: {basename}]\n{code}\n\n"

                        try:
                            exec_result = self.code_manager.run_code(abs_path, code)
                            prompt += f"[Code Result: {basename}]\n{repr(exec_result)}\n\n"
                        except Exception as e_exec:
                            prompt += f"[Code Execution Error: {basename}]\n  - {type(e_exec).__name__}: {e_exec}\n\n"

                    except Exception as e_file:
                        prompt += f"[Per-file Error: {abs_path}]\n  - {type(e_file).__name__}: {e_file}\n\n"

            except Exception as e:
                prompt += f"[Code Generation/Execution Section Error]\n  - {type(e).__name__}: {e}\n\n"


        prompt += (
            "You are a reasoning assistant combining database, vector, file, "
            "web search and other evidence. Based on all the above information, "
            f"provide the best possible answer to the user question: '{question}'.\n"
            "Please:\n"
            "- Use concise language (1–2 natural language sentences for simple question, "
            "3-5 sentences for report type question. If user ask to 'List ... in python list format', use that format)\n"
            "- Do NOT include code blocks, URLs, markdown formatting, or other extra text.\n"
            "- If you think the question is wrong, ambiguous, or cannot be answered based on the information available, say so clearly.\n"
            "Final Answer:"
        )

        while True:
            try:
                response = self.llm_client.call_llm(prompt, model=self.llm_model)
                answer = response.strip()
                break
            except Exception as e:
                print(f"[ERROR] LLM failed to generate final answer: {e}")

        
        cost = self.llm_client.get_and_reset_token_usage()

        ret = {
            "answer": answer,
            "token_usage": cost,
            "final_prompt": prompt
        }
        return ret

    
class PlanningAgent(BaseAgent):

    def __init__(self, dataset: dict, dataset_dir: str, tools: list = [], llm_model: str = "deepseek/deepseek-chat"):
        super().__init__(AGENT_TYPE.PLANNING)

        self.tools = tools
        self.llm_model = llm_model
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

        if "code" in tools:
            code_manager = CodeManager()
            self.registry.register_tool(CodeTool(code_manager))
        else:
            self.tools_desc.pop("CodeTool", None)

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
        self.registry.register_tool(LLMTool(llm_client, model=self.llm_model))

    def _call_tool(self, tool_name: str, input_payload: dict) -> dict:
        entry = {
            "tool_name": tool_name,
            "start_time": time.time(),
            "status": None,
            "execution_time": None,
            "output": None,
            "error": None
        }

        try:
            tool_result = self.registry.call_tool(tool_name, input_payload)
            entry["status"] = tool_result.get("status", "unknown")
            entry["execution_time"] = tool_result.get("execution_time")
            entry["output"] = tool_result.get("output")
            entry["error"] = tool_result.get("error")
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"[ERROR] calling {tool_name}: {entry['error']}")
        finally:
            entry["end_time"] = time.time()
            entry["duration"] = round(entry["end_time"] - entry["start_time"], 3)
        return entry


    def _generate_plan_with_llm(self, question: str) -> list:
        tool_info = "\n".join(
            [f"- {name}:\n{desc}" for name, desc in self.tools_desc.items()]
        )

        prompt = f"""
            You are an intelligent task planning assistant. You need to generate a multi-step execution plan based on the user's question.
            The plan should consist of multiple subtasks, and each subtask should correspond to one of the available tools.
            The final step should produce the final answer by using the LLMTool to integrate the outputs from previous steps and generate the final response.
            Do not use LLMTool until the final step.

            [Available Tools]
            {tool_info}

            [Requirements]
            1. The return format must be a JSON array (list), where each element is a subtask dict.
            2. Each subtask must contain the following fields:
               - tool_name: The name of the tool (must be one of the tools listed above)
               - input: A dictionary of input parameters. If 'query' field depends on the output of previous tasks, leave it as an empty string for now.
               - input_rely: An array indicating which previous task indices this task depends on (e.g., [0,1] means it depends on the outputs of the first two tasks). Even if there is no dependency, this must be an empty array.
               - description: A natural language description of the task, as detailed and clear as possible. This field will be used later as reference.
            3. Do not output any irrelevant content. Only output the complete JSON string.

            [User Question]
            {question}

            Please output only the JSON content.
        """.strip()


        llm_tool = self.registry.get_tool("LLMTool")
        plan = None
        while True:
            try:
                response = llm_tool.run({"query": prompt})
                if response["status"] == "success" and response["output"]:
                    raw_output = response["output"]
                    cleaned = raw_output.strip().strip("`").strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    plan = json.loads(cleaned)
                    assert isinstance(plan, list)
                    # 校验基本字段
                    for step in plan:
                        if not all(k in step for k in ("tool_name", "input", "input_rely", "description")):
                            raise ValueError("Subtask missing required fields")
                    break
            except Exception as e:
                print(f"[PlANNING] generate plan failed: {e}. Retrying...")
                plan = None

        if not plan:
            plan = [
                {
                    "tool_name": "LLMTool",
                    "input": {"query": f"{question}"},
                    "input_rely": [],
                    "description": "Use LLMTool to directly answer the user's question."
                }
            ]
            print("[PlANNING] Using default plan.")
        return plan
    
    def _fill_input_with_rely(self, tool_name: str, input_payload: dict, rely_output: list, disc: str, question:str) -> dict:
        if len(rely_output) == 0:
            return input_payload
        
        context_texts = []
        for out in rely_output:
            try:
                if isinstance(out, (dict, list)):
                    context_texts.append(json.dumps(out, ensure_ascii=False))
                else:
                    context_texts.append(str(out))
            except Exception:
                context_texts.append(str(out))
        context = "\n".join(context_texts)

        if tool_name == "LLMTool":
            query = "Now you are a query agent, you need to answer the quesion based on your knowledge and some context.\n"
            query += f"\n[Question] {question}\n[Goal] {disc}\n[Context]\n{context}\n"
            query += "Please ONLY provide a concise and accurate answer."
            query +=(
                "Please:\n"
                "- Use concise language (1–2 natural language sentences for simple question, "
                "3-5 sentences for report type question. If user ask to 'List ... in python list format', use that format)\n"
                "- Do NOT include code blocks, URLs, markdown formatting, or other extra text.\n"
                "- If you think the question is wrong, ambiguous, or cannot be answered based on the information available, say so clearly.\n"
            )
            input_payload["query"] = query
            return input_payload
        
        if tool_name == "CodeTool":
            generation_task = f"""
                You are a Python data processing assistant.

                You need to generate the input parameters for a CodeTool call.
                The CodeTool executes Python code on a SINGLE local file.

                Constraints:
                1. You must output a JSON object with EXACTLY two fields:
                   - "abs_path": absolute path to ONE csv / txt / json file
                   - "code": valid Python code as a string
                2. In the code:
                   - Use variable `data` to access the loaded file content
                     (DataFrame for csv, string for txt, dict for json).
                   - Store the FINAL output in variable `result`.
                   - Do NOT read files. The file is already loaded in `data`.
                   - Do NOT print anything.
                3. The code should be minimal, correct, and directly executable.

                Context (outputs from previous steps):
                {context}

                Main question:
                {question}

                User goal:
                {disc}

                Return ONLY a JSON object. No code blocks! No markdown! No other explanations!
            """.strip()

            llm_tool = self.registry.get_tool("LLMTool")
            llm_response = llm_tool.run({"query": generation_task})

            if llm_response["status"] == "success" and llm_response["output"]:
                #with open("eval_logs/debug_codetool_prompt.txt", "w", encoding="utf-8") as f:
                #    f.write(generation_task)
                #    f.write("\n\n")
                #    f.write(llm_response["output"])
                try:
                    code_input = json.loads(llm_response["output"].strip())
                    assert "abs_path" in code_input and "code" in code_input
                    input_payload["abs_path"] = code_input["abs_path"]
                    input_payload["code"] = code_input["code"]
                except Exception as e:
                    print(f"[PLANNING] failed to parse refined query for CodeTool, error: {e}, using original input.")
            else:
                print(f"[PLANNING] LLM failed to generate refined query for CodeTool, using original input.")

            return input_payload
        
        tool_desc = self.tools_desc.get(tool_name, "")
        if not tool_desc:
            return input_payload

        generation_task = f"""
            You are a tool input generation agent.

            Your task:
            Based on the tool description, context, and user goal,
            generate the input JSON for calling the tool.

            Tool name:
            {tool_name}

            Tool description:
            {tool_desc}

            Context (outputs from previous steps):
            {context}

            Main question:
            {question}

            User goal:
            {disc}

            Constraints:
            1. Return EXACTLY the JSON.
            2. The JSON keys MUST strictly follow the tool's Args format.
            3. Do NOT include explanations, markdown, or extra text.
            4. No code blocks! Only return the JSON.
        """.strip()

        llm_tool = self.registry.get_tool("LLMTool")
        llm_response = llm_tool.run({"query": generation_task})

        if llm_response["status"] == "success" and llm_response["output"]:
            raw_output = llm_response["output"].strip()
            try:
                tool_input = json.loads(raw_output)

                for k, v in tool_input.items():
                    input_payload[k] = v

            except Exception as e:
                print(
                    f"[PLANNING] Failed to parse tool input for {tool_name}: {e}\n"
                    f"Raw output:\n{raw_output}"
                )
        else:
            print(f"[PLANNING] LLM failed to generate input for {tool_name}, using original input.")

        return input_payload


    def handle_query(self, question: str) -> dict:
        execution_log = []

        print("[PLANNING] Generating execution plan...")
        plan = self._generate_plan_with_llm(question)
        execution_log.append({
            "step": "generate_plan",
            "plan": copy.deepcopy(plan)
        })

        tool_outputs = []
        for step_idx, step in enumerate(plan):
            tool_name = step["tool_name"]
            input_payload = step["input"]
            input_rely = step["input_rely"]
            disc = step.get("description", "")

            rely_output = []
            for rely_idx in input_rely:
                if 0 <= rely_idx < len(tool_outputs):
                    rely_output.append({
                        "tool_name": plan[rely_idx].get("tool_name", ""),
                        "description": plan[rely_idx].get("description", ""),
                        "output": tool_outputs[rely_idx]["output"]
                    })
            input_payload = self._fill_input_with_rely(tool_name, input_payload, rely_output, disc, question)

            print(f"[STEP {step_idx}] Calling tool: {tool_name} with input: {json.dumps(input_payload, ensure_ascii=False)[:400]}")
            tool_result = self._call_tool(tool_name, input_payload)
            tool_outputs.append(tool_result)
            
            execution_log.append({
                "step": step_idx,
                "tool_name": tool_name,
                "input": input_payload,
                "output": tool_result
            })

        final_step = tool_outputs[-1]
        if final_step["status"] == "success" and final_step["output"]:
            answer = final_step["output"]
        else:
            answer = f"[PLANNING] Final step failed: {final_step.get('error')}"
        

        cost = self.registry.get_tool("LLMTool").llm_client.get_and_reset_token_usage()
        ret = {
            "answer": answer,
            "token_usage": cost,
            "execution": execution_log,
        }
        return ret