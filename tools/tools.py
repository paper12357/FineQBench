import time
import os
import traceback


class BaseTool:
    """Unified tool base class"""
    def __init__(self, name: str):
        self.name = name

    def run(self, input_data: dict) -> dict:
        start_time = time.time()
        try:
            output = self._execute(input_data)
            status = "success"
            error = None
        except Exception as e:
            print(f"[ERROR: {self.name}] Error during execution: {e}")
            output = None
            status = "error"
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        end_time = time.time()
        return {
            "status": status,
            "tool_name": self.name,
            "execution_time": round(end_time - start_time, 3),
            "output": output,
            "error": error,
        }

    def _execute(self, input_data: dict):
        raise NotImplementedError


class DBExecuteTool(BaseTool):
    def __init__(self, db_manager):
        super().__init__("DBExecuteTool")
        self.db_manager = db_manager

    def _execute(self, input_data):
        sql = input_data.get("query")
        if not sql:
            raise ValueError("Missing SQL statement.")
        return self.db_manager.query_by_sql(sql)
    
class DBSchemaTool(BaseTool):
    def __init__(self, db_manager):
        super().__init__("DBSchemaTool")
        self.db_manager = db_manager

    def _execute(self, input_data):
        return self.db_manager.get_db_schema()

class VectorTool(BaseTool):
    def __init__(self, vec_manager):
        super().__init__("VectorTool")
        self.vec_manager = vec_manager

    def _execute(self, input_data):
        query_text = input_data.get("query")
        top_k = input_data.get("k", 5)
        return self.vec_manager.search(query_text=query_text, top_k=top_k, raw=True)
    
class LotusTool(BaseTool):
    def __init__(self, lotus_manager):
        super().__init__("LotusTool")
        self.lotus_manager = lotus_manager

    def _execute(self, input_data):
        type = input_data.get("type", "filter")
        table = input_data.get("table")
        query_text = input_data.get("query")
        top_k = input_data.get("k", 5)
        if type == "topk":
            return self.lotus_manager.topk(table_name=table, query_text=query_text, top_k=top_k)
        if type == "filter":
            return self.lotus_manager.filter(table_name=table, query_text=query_text)
        else:
            return self.lotus_manager.filter(table_name=table, query_text=query_text)
        
class GraphTool(BaseTool):
    def __init__(self, graph_manager):
        super().__init__("GraphTool")
        self.graph_manager = graph_manager

    def _execute(self, input_data):
        type = input_data.get("type", "relation")
        entity1 = input_data.get("entity1")
        entity2 = input_data.get("entity2", "")
        if type == "relation":
            return self.graph_manager.search_relation(entity1, entity2)
        if type == "entity":
            return self.graph_manager.search_entity(entity1)
        else:
            return self.graph_manager.search_entity(entity1)

class FileTool(BaseTool):
    def __init__(self, file_manager):
        super().__init__("FileTool")
        self.file_manager = file_manager

    def _execute(self, input_data):
        key = input_data.get("query")
        limit = input_data.get("k", 2)
        file_paths = self.file_manager.fetch_file(key=key, limit=limit)
        results = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            results.append({"abs_path": path, "preview": content})
        return results

class WebTool(BaseTool):
    def __init__(self, search_client):
        super().__init__("WebTool")
        self.search_client = search_client

    def _execute(self, input_data):
        query = input_data.get("query")
        num_results = input_data.get("k", 5)
        return self.search_client.search(query=query, num_results=num_results)


class CodeTool(BaseTool):
    def __init__(self, code_manager):
        super().__init__("CodeTool")
        self.code_manager = code_manager

    def _execute(self, input_data):
        path = input_data.get("abs_path")
        code = input_data.get("code")
        return self.code_manager.run_code(path, code)
    
class LLMTool(BaseTool):
    def __init__(self, llm_client, model="deepseek/deepseek-chat"):
        super().__init__("LLMTool")
        self.llm_client = llm_client
        self.model = model

    def _execute(self, input_data):
        prompt = input_data.get("query")
        model = input_data.get("model", self.model)
        return self.llm_client.call_llm(prompt, model=model)

class VisionTool(BaseTool):
    def __init__(self, vision_client):
        super().__init__("VisionTool")
        self.vision_client = vision_client

    def _execute(self, input_data):
        prompt = input_data.get("prompt")
        image_path = input_data.get("image_path")
        model = input_data.get("model", "gpt-4.1")
        return self.vision_client.call_vision(prompt, image_path, model=model)

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register_tool(self, tool: BaseTool):
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool subclasses can be registered.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool
        print(f"[ToolRegistry] Registered tool: {tool.name}")

    def unregister_tool(self, tool_name: str):
        if tool_name in self._tools:
            del self._tools[tool_name]
            print(f"[ToolRegistry] Unregistered tool: {tool_name}")
        else:
            print(f"[ToolRegistry] Tool '{tool_name}' not found.")

    def list_tools(self) -> list:
        return list(self._tools.keys())

    def get_tool(self, tool_name: str) -> BaseTool:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")
        return self._tools[tool_name]

    def call_tool(self, tool_name: str, input_data: dict) -> dict:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")
        tool = self._tools[tool_name]
        print(f"[ToolRegistry] Calling tool: {tool.name}")
        return tool.run(input_data)