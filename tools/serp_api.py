import os
from serpapi import GoogleSearch

class SearchClient:
    def __init__(self, api_key: str = None, engine: str = "google"):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("plese provide SERPAPI_KEY via parameter or SERPAPI_KEY environment variable.")
        self.engine = engine

    def search(self, query: str, num_results: int = 5, site: str = None, raw: bool = False):        
        if site:
            query = f"{query} site:{site}"

        params = {
            "engine": self.engine,
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "hl": "en"
        }

        data = GoogleSearch(params)
        results = data.get_dict()

        snippets = []
        if "organic_results" in results:
            for item in results["organic_results"][:num_results]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                snippets.append({"title": title, "snippet": snippet, "link": link})
        
        formatted = [
            f"Title: {item['title']}\nSnippet: {item['snippet']}\nLink: {item['link']}"
            for item in snippets
        ]

        return "\n\n".join(formatted)

