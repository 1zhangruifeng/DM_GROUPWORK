import re, json, logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class OfflineEvaluator:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.anomaly_patterns = self._init_anomaly_rules()
        self.VALIDATION_PATTERNS = [
            # =============================
            # 英文部分（完全匹配你现有 agent 风格）
            # =============================

            # --- Emotion Naming ---
            r"I (hear|see|understand) (that|how) (you('|’)re|you are) feeling",
            r"It sounds like (you('|’)re|you are) feeling",
            r"You (seem|sound) (really|so)? ?(upset|anxious|overwhelmed|sad|hurt|frustrated|angry)",
            r"You mentioned that .*? made you feel",
            r"From what you said,.*?you’re feeling",

            # --- Reflective Listening ---
            r"You’re saying that",
            r"It seems like the situation with .*? is weighing on you",
            r"When you said .*?, it really shows how you feel",
            r"The way you described .*? suggests you’re feeling",

            # --- Validation ---
            r"That(’|'| is) completely understandable",
            r"Anyone in your situation would feel this way",
            r"What you’re feeling makes complete sense",
            r"It’s valid to feel this way",
            r"Your feelings are real and reasonable",
            r"It’s okay to feel .*? given what you’ve been through",

            # --- Quoting/Paraphrasing ---
            r"When you mentioned",
            r"As you said",
            r"Based on what you wrote",
            r"From your words",

            # =============================
            # 中文部分（按你的 agent 规则设计）
            # =============================

            # --- 中文情绪命名（必须点名情绪 + 用户语境） ---
            r"我听到你说.*?(难过|痛苦|焦虑|生气|崩溃|压力|伤心|沮丧|不安|愤怒)",
            r"听起来你.*?(难过|痛苦|崩溃|紧张|焦虑|不安|低落|心累)",
            r"从你描述的.*?(事情|情况|经历).*?可以感觉到你很.*?(难受|辛苦|紧绷)",
            r"你刚刚提到.*?让你觉得.*?(难受|害怕|担心|委屈)",
            r"我能感受到你现在的情绪是.*?(很沉重|很难受|很混乱)",

            # --- 中文反映式倾听（Reflective Listening） ---
            r"你刚才说.*?这让我看到你真的很在意",
            r"从你的描述里能感觉到.*?对你影响很大",
            r"当你提到.*?的时候，我能感到你内心的波动",
            r"你说的.*?显然让你很不好受",
            r"你分享的这些让我看到你现在的状态确实不轻松",

            # --- 中文情绪验证（Validation） ---
            r"你有这样的感受是很正常的",
            r"你的感受完全可以理解",
            r"在这样的情况下有.*?的情绪是很合理的",
            r"任何人在这样的情境下都会感到.*?(难受|压力|不安)",
            r"你现在的感受很真实也很重要",
            r"你会这样感觉真的一点都不奇怪",

            # --- 引用用户原话（必须复述/引用用户话语） ---
            r"当你说.*?的时候",
            r"正如你刚刚提到的.*?",
            r"从你刚刚的文字里看到.*?",
            r"你提到.*?这一点真的能感觉到你的处境",
        ]

    def _load_model(self):
        """加载轻量化多语言模型"""
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        logger.info(f"正在加载模型: {model_name}")
        return SentenceTransformer(model_name, device=self.device)

    def _init_anomaly_rules(self) -> Dict[str, List[str]]:
        """异常检测规则库"""
        return {
            "invalidating": ["你不应该", "别想太多", "你太敏感", "坚强点", "这没什么", "想开点"],
            "extreme": ["自杀", "kill yourself", "去死", "报复社会", "砍人", "杀人"],
            "unprofessional": ["他妈的", "傻逼", "你妈的", "去你的", "bitch"]
        }

    def classify_issue_type(self, text: str) -> str:
        """智能识别用户情感问题类型"""
        text_lower = text.lower() if text else ""

        if any(kw in text_lower for kw in
               ["分手", "失恋", "前任", "ex", "离婚", "breakup", "heartbreak", "divorce"]):
            return "romantic breakup"
        elif any(kw in text_lower for kw in ["吵架", "争吵", "冲突", "矛盾", "绝交", "误会", "朋友", "室友", "fight",
                                             "argument", "conflict", "quarrel", "contradiction", "break off relations",
                                             "misunderstanding", "friends", "roommate"]):
            return "interpersonal conflict"
        elif any(kw in text_lower for kw in
                 ["工作", "职场", "老板", "同事", "绩效", "加班", "kpi", "裁员", "work", "job", "career",
                  "workplace", "boss", "colleague", "performance", "overtime", "layoffs"]):
            return "workplace stress"
        elif any(kw in text_lower for kw in
                 ["焦虑", "抑郁", "压力", "失眠", "情绪", "心理", "难受", "anxiety", "depressed", "stress",
                  "insomnia", "emotion", "psychology", "discomfort"]):
            return "mental health"
        elif any(kw in text_lower for kw in
                 ["家人", "家庭", "父母", "亲戚", "沟通", "代沟", "family", "parents", "relatives",
                  "communication", "generation gap"]):
            return "family issues"
        elif any(kw in text_lower for kw in
                 ["钱", "经济", "贫穷", "债务", "买不起", "money", "economy", "poverty", "debt", "unaffordable"]):
            return "financial stress"
        elif any(kw in text_lower for kw in
                 ["考试", "挂科", "学习", "学业", "论文", "毕业", "gpa", "成绩", "exam", "fail", "study",
                  "academic performance", "thesis", "graduation", "gpa", "grade"]):
            return "academic anxiety"
        else:
            return "general emotional distress"

    def preprocess(self, text: str) -> str:
        """快速预处理"""
        if not text: return ""
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text.lower())
        return ' '.join(text.split())

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        """获取语义向量"""
        processed = [self.preprocess(t) for t in texts]
        return self.model.encode(processed, convert_to_tensor=True, device=self.device).cpu().numpy()

    def detect_emotion(self, text: str) -> str:
        """增强版情绪检测：支持更广泛的共情表达"""
        text_lower = text.lower()

        # 共情标记库（大幅扩展，覆盖多种表达方式）
        empathy_markers = {
            "悲伤": [
                "理解你的难过", "听到你伤心", "感受到你的痛苦", "心碎很正常", "悲伤是可以理解的", "难过是正常的",
                "sad", "heartbroken", "upset", "grief", "cry", "tears", "depressed", "devastated"
            ],
            "焦虑": [
                "理解你的焦虑", "感受到你的压力", "担心是正常的", "紧张是可以理解的", "焦虑是自然的",
                "anxious", "worried", "stress", "nervous", "overwhelmed", "panic"
            ],
            "愤怒": [
                "理解你的愤怒", "生气是正常的", "有权感到气愤", "愤怒是合理的反应",
                "angry", "frustrated", "mad", "furious", "irritated", "annoyed"
            ]
        }

        # 1. 先检测是否包含共情回应（Agent回复）
        for emotion, markers in empathy_markers.items():
            if any(marker in text_lower for marker in markers):
                return emotion

        # 2. 检测直接情绪表达（用户输入）
        user_emotions = {
            "悲伤": ["难过", "伤心", "痛苦", "失望", "心碎", "悲伤", "哭泣", "难受", "失落", "心累"],
            "焦虑": ["焦虑", "担心", "压力", "紧张", "不安", "害怕", "恐惧", "纠结", "紧绷", "心烦"],
            "愤怒": ["生气", "愤怒", "恼火", "气愤", "恼怒", "不爽", "火大", "激怒"]
        }

        for emotion, keywords in user_emotions.items():
            if any(kw in text_lower for kw in keywords):
                return emotion

        return "neutral"

    def contains_emotion_validation(self, text: str) -> bool:
        for pattern in self.VALIDATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def compute_all_metrics(self, data: List[Dict[str, str]]) -> Dict[str, float]:
        """计算所有核心指标"""
        if len(data) < 3:
            return {"error": "数据量不足（至少需要3轮对话）"}

        user_inputs = [d["user_input"] for d in data]
        responses = [d["agent_response"] for d in data]

        # 1. 语义相似度
        user_emb = self.get_embedding(user_inputs)
        resp_emb = self.get_embedding(responses)
        sims = [float(cosine_similarity([u], [r])[0][0]) for u, r in zip(user_emb, resp_emb)]
        avg_sim = np.mean(sims)

        # 2. 情绪匹配度
        success_count = 0
        empathy_markers = [
            "我能感受到你", "理解你的", "听到你", "感受到你的", "我看到你", "你现在的情绪", "你的感受"
        ]

        for ui, resp in zip(user_inputs, responses):
            user_type = self.classify_issue_type(ui)
            resp_lower = resp.lower()

            # 情感问题类型匹配
            agent_type_match = False
            if user_type == self.classify_issue_type(resp):
                agent_type_match = True

            # 共情表达匹配
            empathy_match = any(marker in resp for marker in empathy_markers)

            if agent_type_match or empathy_match:
                success_count += 1

        alignment = success_count / len(user_inputs)

        # 3. 冗余度
        ngram_counts = {}
        for resp in responses:
            words = resp.split()
            for a, b in zip(words, words[1:]):
                ngram_counts[(a, b)] = ngram_counts.get((a, b), 0) + 1
        redundancy = sum(c - 1 for c in ngram_counts.values() if c > 1) / max(len(ngram_counts), 1)

        # 4. 异常率
        anomalies = 0
        for resp in responses:
            resp_text = resp.lower()
            if any(p in resp_text for patterns in self.anomaly_patterns.values() for p in patterns):
                anomalies += 1
        anomaly_rate = anomalies / len(responses)

        # 5. 场景覆盖率
        # 简化版：统计不同输入类型的覆盖情况
        issues = set()
        for ui in user_inputs:
            if any(k in ui for k in ["分手", "失恋", "ex", "heartbreak"]):
                issues.add("breakup")
            elif any(k in ui for k in ["吵架", "冲突", "绝交", "argument", "conflict"]):
                issues.add("conflict")
            elif any(k in ui for k in ["工作", "职场", "绩效", "work", "job", "career"]):
                issues.add("work")
            elif any(k in ui for k in ["考试", "挂科", "学业", "exam", "academic"]):
                issues.add("academic")
            elif any(k in ui for k in ["家人", "父母", "family", "parents"]):
                issues.add("family")
            elif any(k in ui for k in ["钱", "经济", "债务", "money", "debt"]):
                issues.add("financial")
            elif any(k in ui for k in ["焦虑", "抑郁", "stress", "anxiety"]):
                issues.add("mental")
            else:
                issues.add("general")
        total_possible_scenarios = min(7, len(set(
            [self.classify_issue_type(ui) for ui in user_inputs]
        )))
        coverage = len(issues) / max(total_possible_scenarios, 1)

        return {
            "语义相似度": float(avg_sim),
            "情绪匹配度": float(alignment),
            "回复冗余度": float(redundancy),
            "异常输出率": float(anomaly_rate),
            "场景覆盖率": float(coverage),
            "对话轮次": len(data)
        }

    def generate_report(self, metrics: Dict[str, float], output_path: str = None) -> str:
        """生成评估报告"""
        rounds = metrics.get('对话轮次', 0)
        # 动态情绪匹配度目标
        if rounds <= 5:
            emotion_target = 0.6
        elif rounds <= 20:
            emotion_target = 0.75
        else:
            emotion_target = 0.8

        report = f"""
评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
评估轮次: {rounds}

{'=' * 60}
核心指标
{'=' * 60}
语义相似度:  {metrics.get('语义相似度', 0):.3f}  |  目标>0.70  |  {'✅' if metrics.get('语义相似度', 0) > 0.7 else '❌'}
情绪匹配度:  {metrics.get('情绪匹配度', 0):.3f}  |  目标>{emotion_target:.2f}  |  {'✅' if metrics.get('情绪匹配度', 0) > emotion_target else '❌'}
回复冗余度:  {metrics.get('回复冗余度', 0):.3f}  |  目标<0.20  |  {'✅' if metrics.get('回复冗余度', 0) < 0.2 else '❌'}
异常输出率:  {metrics.get('异常输出率', 0):.3f}  |  目标<0.001 |  {'✅' if metrics.get('异常输出率', 0) < 0.001 else '❌'}
场景覆盖率:  {metrics.get('场景覆盖率', 0):.3f}  |  目标>0.70  |  {'✅' if metrics.get('场景覆盖率', 0) > 0.7 else '❌'}

{'=' * 60}
改进建议
{'=' * 60}
"""

        if metrics.get('语义相似度', 0) < 0.7:
            report += "• 语义相似度低：Agent回复偏离用户意图，需优化prompt清晰度\n"
        if metrics.get('情绪匹配度', 0) < emotion_target:
            report += "• 情绪匹配度低：Empathy Agent需要更强的情绪识别能力\n"
        if metrics.get('回复冗余度', 0) > 0.2:
            report += "• 冗余度高：四个Agent回复重复内容过多，需差异化指令\n"
        if metrics.get('异常输出率', 0) > 0.001:
            report += "• 异常率超标：立即检查agent.py，添加异常内容过滤规则\n"

        if all([
            metrics.get('语义相似度', 0) > 0.7,
            metrics.get('情绪匹配度', 0) > emotion_target,
            metrics.get('回复冗余度', 0) < 0.2,
            metrics.get('异常输出率', 0) < 0.001
        ]):
            report += "✅ 所有核心指标均已达标！\n"

        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
        return report


def run_evaluation(history_file: str = "conversation_history.json", output_dir: str = "evaluation_logs"):
    """主评估流程"""
    try:
        data = json.loads(Path(history_file).read_text(encoding='utf-8'))
        conversations = [
            {"user_input": item["input"], "agent_response": item["response"]}
            for item in data[-50:]
            if item.get("response") and item.get("input")
        ]
        if not conversations:
            logger.error("没有找到有效的对话数据")
            return
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    evaluator = OfflineEvaluator()
    metrics = evaluator.compute_all_metrics(conversations)

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_file = Path(output_dir) / f"metrics_{timestamp}.json"
    result_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    report_file = Path(output_dir) / f"report_{timestamp}.txt"
    report = evaluator.generate_report(metrics, str(report_file))

    logger.info(f"✓ 评估完成，报告保存至: {report_file}")
    logger.info(f"✓ 详细指标保存至: {result_file}")
    print(report)


if __name__ == "__main__":
    run_evaluation()
