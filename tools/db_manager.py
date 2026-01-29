import sqlite3
import os

class DBManager:
    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"db file not found: {db_path}")
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        print(f"connected to database: {db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("closed database connection")

    def list_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in self.cursor.fetchall()]
        tables = [t for t in tables if t != "sqlite_sequence"]
        return tables

    def list_fields(self, table_name: str):
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        fields = self.cursor.fetchall()
        return fields

    def query_all(self, table_name: str, limit: int = 10):
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return rows

    def query_by_sql(self, sql: str):
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return rows
    
    def get_db_schema(self) -> str:
        prompt = ""
        tables = self.list_tables()
        for table in tables:
            prompt += f"Table: {table}\n"
            fields = self.list_fields(table)
            prompt += "Fields:\n"
            for field in fields:
                prompt += f"  - {field[1]} ({field[2]})\n"
            sample_data = self.query_all(table, limit=20)
            prompt += "Sample Data:\n"
            for row in sample_data:
                prompt += f"  - {row}\n"
            prompt += "\n"
        return prompt