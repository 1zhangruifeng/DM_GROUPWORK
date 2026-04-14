import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


class VectorIndex:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.knowledge_base = []
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def build_index(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base

        texts = [item['content'] for item in knowledge_base]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        quantizer = faiss.IndexFlatL2(self.dimension)
        nlist = min(100, max(1, len(knowledge_base) // 10))
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

        self.index.train(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))

        print(f"索引构建完成，共 {self.index.ntotal} 条知识")

    def search(self, query: str, k: int = 5, issue_type: str = None) -> List[Dict]:
        query_emb = self.embedding_model.encode([query])

        distances, indices = self.index.search(query_emb.astype('float32'), min(k * 2, self.index.ntotal))

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.knowledge_base):
                continue

            item = self.knowledge_base[idx]

            if issue_type and item.get('issue_type'):
                if item.get('issue_type') != issue_type:
                    continue

            results.append({
                'content': item['content'],
                'title': item['title'],
                'source': item['source'],
                'score': float(1 / (1 + dist)),
                'type': item.get('type', 'unknown')
            })

        return results[:k]

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.knowledge_base, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", 'rb') as f:
            self.knowledge_base = pickle.load(f)