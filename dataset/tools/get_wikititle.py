import time
from tools.http_request import safe_get
from tools.wiki_api import  *

input_file = "entity_data/geography.txt"
output_file = "entity_data/geography_wiki.txt"

def search_wiki(name):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "format": "json",
        "srlimit": 1
    }
    r = safe_get(WIKI_API, params=params, headers=HEADERS)
    data = r.json()
    try:
        title = data["query"]["search"][0]["title"]
        return title
    except (IndexError, KeyError):
        return name

with open(input_file, "r", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

wiki_titles = []

for idx, name in enumerate(names, start=1):
    wiki_title = search_wiki(name)
    wiki_titles.append(wiki_title)
    print(f"{idx}/{len(names)}: {name} -> {wiki_title}")
    time.sleep(0.1) 

with open(output_file, "w", encoding="utf-8") as f:
    for t in wiki_titles:
        f.write(t + "\n")
