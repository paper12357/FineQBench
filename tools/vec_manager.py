import chromadb
from sentence_transformers import SentenceTransformer


class VectorSearchManager:

    def __init__(self, db_path: str, collection_name: str,
                 model_name: str):

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def search(self, query_text: str, top_k: int = 5, raw: bool = False):
        if not query_text.strip():
            return []

        query_emb = self.model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        output = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][i]
            output.append({
                "source": metadata.get("source", "unknown"),
                "dir": metadata.get("dir", "unknown"),
                "paragraph": doc
            })
        if raw:
            return output
        return "\n\n".join(
            [f"Source: {item['source']} (Dir: {item['dir']})\nParagraph: {item['paragraph']}"
             for item in output]
        )

