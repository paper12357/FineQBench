import pandas as pd
import lotus
import os,json
from lotus.models import LM

class LotusManager:

    def __init__(self, db_path: str):
        lm = LM(model="gpt-4.1-nano")
        lotus.settings.configure(lm=lm)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"lotus data json not found: {db_path}")

        with open(db_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.tables = {} 

        for table_name, paragraphs in raw_data.items():
            if not isinstance(paragraphs, list):
                continue
            df = pd.DataFrame({table_name: paragraphs})
            self.tables[table_name] = df

    def get_table_names(self) -> str:
        return ', '.join(self.tables.keys())

    def topk(self, table_name: str, query_text: str, top_k: int = 5):
        res = self.tables[table_name].sem_topk(query_text, top_k)
        list = res[table_name].tolist()
        list = [item.replace('\n', ' ').strip() for item in list]
        list = [item if len(item)<=200 else item[:200]+'...' for item in list]
        return '\n\n'.join(list)
        
    def filter(self, table_name: str, query_text: str):
        res = self.tables[table_name].sem_filter(query_text)
        list = res[table_name].tolist()
        list = [item.replace('\n', ' ').strip() for item in list]
        list = [item if len(item)<=200 else item[:200]+'...' for item in list]
        return '\n\n'.join(list)
    
    def join(self, left_table_name: str, right_table_name: str, query_text: str):
        res = self.tables[left_table_name].sem_join(self.tables[right_table_name], query_text)
        return res

        


