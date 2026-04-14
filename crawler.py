import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
import time


class PsychologyCrawler:

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def crawl_zhihu(self, keyword: str, pages: int = 5) -> List[Dict]:
        results = []
        for page in range(pages):
            url = f"https://www.zhihu.com/api/v4/search_v3?q={keyword}&page={page}"
            response = requests.get(url, headers=self.headers)
            data = response.json()

            for item in data.get('data', []):
                results.append({
                    'source': 'zhihu',
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'url': item.get('url', ''),
                    'type': 'qa'
                })
            time.sleep(1)
        return results

    def crawl_baike(self, concept: str) -> Dict:
        url = f"https://baike.baidu.com/item/{concept}"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        content = soup.find('div', class_='main-content')

        return {
            'source': 'baike',
            'title': concept,
            'content': content.get_text() if content else '',
            'type': 'concept'
        }

    def crawl_psychology_today(self, topic: str) -> List[Dict]:
        url = f"https://www.psychologytoday.com/us/search?keys={topic}"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = []
        for article in soup.find_all('article', limit=20):
            articles.append({
                'source': 'psychology_today',
                'title': article.find('h2').get_text() if article.find('h2') else '',
                'content': article.find('p').get_text() if article.find('p') else '',
                'type': 'article'
            })
        return articles


KEYWORDS = [
    "情感支持", "共情技巧", "认知行为疗法", "情绪管理",
    "人际关系修复", "焦虑缓解", "自我关怀", "心理韧性",
    "conflict resolution", "empathy training", "CBT techniques"
]