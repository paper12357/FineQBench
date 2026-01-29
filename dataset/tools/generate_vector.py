import os
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="dbvec")
collection = client.get_or_create_collection("football")

model = SentenceTransformer("models/all-MiniLM-L6-v2")


def split_into_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n\n") if len(p.strip()) > 30]
    return paragraphs


data_dirs = ["football_players", "football_clubs"]

for data_dir in data_dirs:
    if not os.path.exists(data_dir):
        continue

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        paragraphs = split_into_paragraphs(text)
        ids = [f"{filename}::p{i}" for i in range(len(paragraphs))]
        embeddings = model.encode(paragraphs).tolist()
        metadatas = [{"source": filename.replace(".txt", ""), "dir": data_dir} for _ in paragraphs]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=paragraphs,
            metadatas=metadatas
        )

