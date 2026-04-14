import json
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class KnowledgeBaseBuilder:

    def __init__(self):
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunk_size = 500

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        if len(text) < 20:
            return ""
        return text.strip()

    def chunk_text(self, text: str, title: str) -> List[Dict]:
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append({
                    'title': title,
                    'content': chunk,
                    'start_idx': i,
                    'end_idx': min(i + self.chunk_size, len(words))
                })
        return chunks

    def build_from_crawled_data(self, crawled_files: List[str]) -> List[Dict]:
        knowledge_base = []

        for file_path in crawled_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                clean_content = self.clean_text(item['content'])
                if not clean_content:
                    continue

                chunks = self.chunk_text(clean_content, item['title'])

                for chunk in chunks:
                    knowledge_base.append({
                        'id': len(knowledge_base),
                        'title': item['title'],
                        'content': chunk['content'],
                        'source': item.get('source', 'unknown'),
                        'type': item.get('type', 'article'),
                        'url': item.get('url', ''),
                        'issue_type': self.classify_content(item['title'] + item['content'])
                    })

        return knowledge_base

    def classify_content(self, text: str) -> str:
        text_lower = text.lower()

        categories = {
            'breakup': ['分手', '失恋', '前任', 'breakup', 'heartbreak'],
            'conflict': ['吵架', '冲突', '沟通', 'conflict', 'argument'],
            'anxiety': ['焦虑', '压力', 'anxiety', 'stress'],
            'depression': ['抑郁', '情绪低落', 'depression', 'sad'],
            'work': ['工作', '职场', '绩效', 'work', 'career'],
            'family': ['家人', '父母', '家庭', 'family', 'parent']
        }

        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category

        return 'general'