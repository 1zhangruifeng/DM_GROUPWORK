import streamlit as st
from agent import build_agents, ModelChoice
from utils import process_images, logger
from agno.media import Image as AgnoImage
from agno.exceptions import ModelProviderError
from pathlib import Path
import tempfile
import os
import json
from datetime import datetime
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict

st.set_page_config(page_title="Emotional Recovery AI Assistant", page_icon="😀", layout="wide")

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "gemini"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "enable_rag" not in st.session_state:
    st.session_state.enable_rag = True


class RAGKnowledgeBase:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.knowledge_base = []
        self.is_ready = False

    def load_or_create(self):
        index_path = Path("./knowledge_base/psychology_index")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path.with_suffix('.faiss').exists() and index_path.with_suffix('.pkl').exists():
            try:
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.index = faiss.read_index(str(index_path.with_suffix('.faiss')))
                with open(index_path.with_suffix('.pkl'), 'rb') as f:
                    self.knowledge_base = pickle.load(f)
                self.is_ready = True
                return True
            except Exception as e:
                st.warning(f"加载知识库失败: {e}")
                return False
        else:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            dimension = 384
            self.index = faiss.IndexFlatL2(dimension)
            self.is_ready = True
            return True

    def search(self, query: str, issue_type: str = None, k: int = 3) -> List[Dict]:
        if not self.is_ready or self.index.ntotal == 0:
            return []

        query_emb = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_emb.astype('float32'), min(k * 2, self.index.ntotal))

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.knowledge_base):
                continue

            item = self.knowledge_base[idx]

            if issue_type and item.get('issue_type'):
                if item['issue_type'] != issue_type and item['issue_type'] != 'general':
                    continue

            results.append({
                'content': item['content'],
                'title': item['title'],
                'source': item.get('source', '未知'),
                'score': float(1 / (1 + dist)),
                'type': item.get('type', 'article')
            })

            if len(results) >= k:
                break

        return results

    def add_knowledge(self, title: str, content: str, source: str = "manual", issue_type: str = "general"):
        if not self.is_ready:
            self.load_or_create()

        chunks = self._chunk_text(content, title)

        for chunk in chunks:
            emb = self.embedding_model.encode([chunk['content']])
            self.index.add(emb.astype('float32'))
            self.knowledge_base.append({
                'id': len(self.knowledge_base),
                'title': title,
                'content': chunk['content'],
                'source': source,
                'issue_type': issue_type,
                'type': 'manual'
            })

        self._save()

    def _chunk_text(self, text: str, title: str, chunk_size: int = 500) -> List[Dict]:
        chunks = []
        words = text.split()

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk and len(chunk) > 20:
                chunks.append({'title': title, 'content': chunk})

        if not chunks and text:
            chunks.append({'title': title, 'content': text[:1000]})

        return chunks

    def _save(self):
        index_path = Path("./knowledge_base/psychology_index")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path.with_suffix('.faiss')))
        with open(index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.knowledge_base, f)


def init_builtin_knowledge(rag):
    if rag.index.ntotal > 0:
        return

    builtin_knowledge = [
        {"title": "有效共情的四个步骤",
         "content": "有效共情包含四个关键步骤：1. 倾听而不评判，让对方充分表达；2. 识别并命名情绪，如'听起来你感到很沮丧'；3. 验证情绪的合理性，让对方知道'有这种感觉是完全正常的'；4. 表达理解和支持。共情的核心是让对方感到被看见、被理解。",
         "source": "心理学知识库", "issue_type": "general"},
        {"title": "认知行为疗法 - 识别思维扭曲",
         "content": "常见的思维扭曲类型：1. 非黑即白：把事情看成全好或全坏；2. 过度概括：把一次失败看成永远会失败；3. 灾难化：总是预期最坏的结果；4. 个人化：把所有问题都归咎于自己；5. 读心术：自以为知道别人在想什么。识别这些思维模式是认知重构的第一步。",
         "source": "心理学知识库", "issue_type": "general"},
        {"title": "人际关系冲突解决技巧",
         "content": "解决人际冲突的有效方法：1. 使用'我'语句表达感受，避免指责对方；2. 积极倾听，先理解对方的立场再表达自己；3. 寻找共同目标，而不是争论谁对谁错；4. 给彼此冷静的时间；5. 关注解决方案而不是追究责任。",
         "source": "心理学知识库", "issue_type": "interpersonal conflict"},
        {"title": "工作压力管理策略",
         "content": "职场压力管理实用策略：1. 设置明确的工作边界，学会说'不'；2. 分解大任务为小步骤，降低焦虑感；3. 定期进行短暂休息；4. 建立支持系统，与信任的同事交流；5. 区分可控和不可控因素，专注于能改变的事情。",
         "source": "心理学知识库", "issue_type": "workplace stress"},
        {"title": "分手后情绪恢复指南",
         "content": "分手后的情绪恢复是一个过程：1. 允许自己感受悲伤、愤怒等情绪；2. 建立新的日常规律；3. 重新发现自己的身份和价值；4. 与朋友和家人保持联系；5. 给自己时间，不要急于开始新的恋情。",
         "source": "心理学知识库", "issue_type": "romantic breakup"},
        {"title": "焦虑缓解的接地技巧",
         "content": "5-4-3-2-1接地技巧：说出你能看到的5样东西；触摸你能摸到的4样东西；注意你能听到的3种声音；闻到你周围的2种气味；说出你能尝到的1种味道。这个技巧能帮助你将注意力从焦虑转移到当下。",
         "source": "心理学知识库", "issue_type": "mental health"},
        {"title": "学业焦虑应对方法",
         "content": "应对考试焦虑的方法：1. 制定现实的学习计划；2. 使用番茄工作法；3. 练习自我对话，用'我已经尽力准备了'替代'我肯定会考砸'；4. 考试前做深呼吸练习；5. 接受适度的焦虑是正常的。",
         "source": "心理学知识库", "issue_type": "academic anxiety"},
        {"title": "家庭沟通改善技巧",
         "content": "改善家庭沟通的方法：1. 选择合适的沟通时机；2. 表达感受而非指责；3. 尝试理解对方的出发点；4. 建立家庭会议制度；5. 必要时寻求专业帮助。改善家庭关系需要时间和耐心。",
         "source": "心理学知识库", "issue_type": "family issues"},
        {"title": "财务压力心理调适",
         "content": "应对财务压力的心理策略：1. 区分'需要'和'想要'；2. 制定可行的预算计划；3. 避免与他人比较；4. 学习基本理财知识；5. 记住金钱不是自我价值的唯一衡量标准。",
         "source": "心理学知识库", "issue_type": "financial stress"}
    ]

    for item in builtin_knowledge:
        rag.add_knowledge(
            title=item["title"],
            content=item["content"],
            source=item["source"],
            issue_type=item["issue_type"]
        )

    st.success("已加载内置心理学知识库！")


@st.cache_resource
def init_rag():
    rag = RAGKnowledgeBase()
    if rag.load_or_create():
        init_builtin_knowledge(rag)
    return rag


def classify_issue_type(text: str) -> str:
    text_lower = text.lower() if text else ""

    if any(kw in text_lower for kw in
           ["分手", "失恋", "前任", "ex", "离婚", "Breakup", "heartbreak", "divorce"]):
        return "romantic breakup"
    elif any(kw in text_lower for kw in ["吵架", "争吵", "冲突", "矛盾", "绝交", "误会", "朋友", "室友", "fight",
                                         "argument", "conflict", "quarrel", "contradiction", "Break off relations",
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
    elif any(k in text_lower for k in
             ["钱", "经济", "贫穷", "债务", "买不起", "money", "economy", "poverty", "debt", "unaffordable"]):
        return "financial stress"
    elif any(k in text_lower for k in
             ["考试", "挂科", "学习", "学业", "论文", "毕业", "gpa", "成绩", "exam", "fail", "study",
              "academic", "thesis", "graduation", "grade"]):
        return "academic anxiety"
    else:
        return "general emotional distress"


def save_history():
    try:
        Path("conversation_history.json").write_text(
            json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    except Exception as e:
        logger.error(f"Failed to save the history record: {e}")


with st.sidebar:
    st.header("Chat History")
    if not st.session_state.history:
        st.info("Your past submissions will appear here.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.container(border=True):
                st.markdown(f"**{len(st.session_state.history) - i}:** {item['input'][:40]}...")
                if item.get('files'):
                    st.caption(f" {len(item['files'])} files")
                if item.get('issue_type'):
                    st.caption(f" {item['issue_type']}")

    st.markdown("---")
    st.session_state.enable_rag = st.checkbox(
        " Enable Knowledge Base (RAG)",
        value=st.session_state.enable_rag,
        help="Enable to retrieve psychology knowledge for better responses"
    )
    st.markdown("---")
    st.markdown("""<div style='text-align:center'><p>Created by Data Mining Group</p>
    <p>We sincerely hope that you can mend your relationship here</p></div>""", unsafe_allow_html=True)

center_col, right_col = st.columns([0.7, 0.3])

with right_col:
    with st.container(border=True):
        st.header("Configuration")

        model_choice: ModelChoice = st.selectbox(
            "Choose your model",
            options=["gemini", "openai", "claude", "deepseek"],
            index=["gemini", "openai", "claude", "deepseek"].index(st.session_state.model_choice),
            help="Select the model you want to use, then enter the corresponding API key below."
        )
        if model_choice != st.session_state.model_choice:
            st.session_state.model_choice = model_choice

        api_key = st.text_input(
            f"Enter {model_choice.upper()} API Key",
            value=st.session_state.api_key,
            type="password",
            help=f"Get your key from the official website of {model_choice.upper()}"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key

        if api_key:
            st.success("API Key provided! ")
        else:
            st.warning("Please enter your API key")
            links = {
                "gemini": "https://makersuite.google.com/app/apikey",
                "openai": "https://platform.openai.com/api-keys",
                "claude": "https://console.anthropic.com/settings/keys",
                "deepseek": "https://platform.deepseek.com/api-keys"
            }
            st.markdown(f"""
            To get your API key:
            please visit: [{model_choice.upper()} Official]({links[model_choice]})
            """)

with center_col:
    st.title("Emotional Recovery AI Assistant")
    st.markdown("""### Your personal emotional recovery AI assistant is here to help you!
    Share your feelings and images, and receive evidence-based support tailored to your situation.""")
    st.divider()

    with st.container(border=True):
        st.subheader("Share Your Feelings")
        user_input = st.text_area("How are you feeling? What happened?", height=150,
                                  placeholder="Tell us your story...", label_visibility="collapsed")

        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            uploaded_files = st.file_uploader("Upload Chat Screenshots (optional)",
                                              type=["jpg", "jpeg", "png"],
                                              accept_multiple_files=True)
        with col2:
            submit_button = st.button("Get Emotional Assistance", type="primary", use_container_width=True,
                                      help="Click to generate your recovery plan")

        if uploaded_files:
            with st.expander("View Uploaded Images"):
                for file in uploaded_files:
                    st.image(file, caption=file.name, width='stretch')

    if submit_button:
        if not st.session_state.api_key:
            st.warning("Please enter your API key in the configuration panel on the right!")
            st.stop()

        if not user_input and not uploaded_files:
            st.warning("Please share your feelings or upload screenshots to get help.")
            st.stop()
        if st.session_state.model_choice == "deepseek" and uploaded_files:
            import cv2
            import numpy as np


            def ocr_image(file):
                img = Image.open(io.BytesIO(file.read()))
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                binary = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 8)
                h, w = binary.shape
                binary = cv2.resize(binary, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                file.seek(0)
                return pytesseract.image_to_string(binary, lang="chi_sim+eng")


            ocr_texts = []
            for file in uploaded_files:
                text = ocr_image(file)
                ocr_texts.append(f"【Image {file.name}】\n{text}")
            user_input = "\n\n".join(ocr_texts) + "\n\n" + (user_input or "")

            with st.expander("OCR Raw Results (Debug)"):
                st.text("\n".join(ocr_texts))

        try:
            agents = build_agents(st.session_state.api_key, st.session_state.model_choice)
            if not all(agents):
                st.error("Failed to initialize agents. Check API key and model choice.")
                st.stop()
            empathy, cognitive, behavioral, motivational = agents

        except Exception as e:
            st.error(f"Failed to build agents: {e}. Please check your API key.")
            logger.error(f"Agent build error: {e}")
            st.stop()

        all_images = process_images(uploaded_files) if uploaded_files else []
        issue_type = classify_issue_type(user_input)

        if st.session_state.model_choice == "deepseek":
            all_images = []

        rag = init_rag() if st.session_state.enable_rag else None

        st.divider()
        st.header(" Your Personalized Recovery Plan")


        def get_rag_context(rag, query: str, issue_type: str):
            if not rag or not st.session_state.enable_rag:
                return "", []

            retrieved = rag.search(query, issue_type=issue_type, k=3)

            if not retrieved:
                return "", []

            context_text = "\n\n".join([
                f"【Reference {i + 1}】Source: {item['source']}\nTitle: {item['title']}\nContent: {item['content'][:500]}..."
                for i, item in enumerate(retrieved)
            ])

            return context_text, retrieved


        def safe_run_with_rag(agent, prompt_template, user_input, issue_type, images, agent_name, rag):
            rag_context, retrieved_items = get_rag_context(rag, user_input, issue_type)

            if rag_context and st.session_state.enable_rag:
                prompt = prompt_template.format(
                    user_input=user_input,
                    issue_type=issue_type,
                    rag_context=rag_context
                )
            else:
                prompt = prompt_template.format(
                    user_input=user_input,
                    issue_type=issue_type,
                    rag_context="(No reference materials available)"
                )

            try:
                response = agent.run(input=prompt, images=images).content
                return response, retrieved_items
            except ModelProviderError as e:
                if "Insufficient Balance" in str(e) or "quota" in str(e).lower():
                    st.error(
                        f" **{st.session_state.model_choice.upper()} account balance is insufficient!**\n\n"
                        f"Please recharge or switch to another model."
                    )
                else:
                    st.error(f"Model call failed (ModelProviderError): {e}")
                logger.error(f"ModelProviderError: {e}")
                st.stop()
            except Exception as e:
                logger.error(f"Agent run error: {e}")
                st.error(f"An exception occurred when generating content: {e}")
                st.stop()


        prompt_empathy_template = """YOUR TASK - EMOTIONAL VALIDATION:

【Psychology Knowledge Base Reference】(RAG retrieved content):
{rag_context}

User's Situation ({issue_type}): "{user_input}"

MANDATORY STEPS:
1. Quote or paraphrase a specific part of their message
2. State their emotion explicitly: "I understand you're feeling [emotion]..."
3. Validate WHY this emotion makes sense in THEIR context
4. Share ONE brief relatable experience about {issue_type}
5. End with personalized encouragement using THEIR words

CRITICAL: If RAG context is provided, use it to support your response. Your response must reference their specific situation, not generic platitudes."""

        prompt_cognitive_template = """YOUR TASK - COGNITIVE RESTRUCTURING:

【Psychology Knowledge Base Reference】(RAG retrieved content):
{rag_context}

User's Challenge ({issue_type}): "{user_input}"

REQUIRED APPROACH:
1. Identify 1-2 specific thought distortions in THEIR story (quote their words)
2. Explain how THEIR specific thinking pattern is unhelpful
3. Offer 2 alternative perspectives tailored to {issue_type}
4. Use Socratic questions referencing THEIR situation

FORBIDDEN: Generic CBT theory without connection to their story."""

        prompt_behavioral_template = """YOUR TASK - ACTIONABLE PLAN:

【Psychology Knowledge Base Reference】(RAG retrieved content):
{rag_context}

User's Context ({issue_type}): "{user_input}"

CREATE A 7-DAY PLAN SPECIFIC TO THEIR SITUATION:
Day 1-2: Immediate coping for THEIR specific stressors
Day 3-4: Activities that address THEIR pain points
Day 5-6: Social media boundaries for {issue_type}
Day 7: Reflection on THEIR progress

RULE: Every suggestion must connect to details in their story. If RAG context is provided, incorporate evidence-based strategies. No generic advice."""

        prompt_motivational_template = """YOUR TASK - PERSONALIZED MOTIVATION:

【Psychology Knowledge Base Reference】(RAG retrieved content):
{rag_context}

User's Struggle ({issue_type}): "{user_input}"

REQUIRED STRUCTURE:
1. Reference THEIR past resilience (ask: what have they overcome?)
2. Connect THEIR strength to THIS specific challenge
3. Use THEIR words to show deep understanding
4. Provide 3 encouraging next steps for THEIR situation

ABSOLUTELY NO generic motivational quotes. Make it deeply personal."""

        all_retrieved = {}

        with st.spinner("Analyzing your emotional state..."):
            st.subheader(" Emotional Validation & Support")
            resp_empathy, retrieved_empathy = safe_run_with_rag(
                empathy, prompt_empathy_template, user_input, issue_type,
                all_images, "Empathy Agent", rag
            )
            st.markdown(resp_empathy)
            all_retrieved['empathy'] = retrieved_empathy

        with st.spinner("Identifying thought patterns..."):
            st.subheader(" Cognitive Restructuring")
            resp_cognitive, retrieved_cognitive = safe_run_with_rag(
                cognitive, prompt_cognitive_template, user_input, issue_type,
                all_images, "Cognitive Agent", rag
            )
            st.markdown(resp_cognitive)
            all_retrieved['cognitive'] = retrieved_cognitive

        with st.spinner("Creating action plan..."):
            st.subheader(" Practical Coping Strategies")
            resp_behavioral, retrieved_behavioral = safe_run_with_rag(
                behavioral, prompt_behavioral_template, user_input, issue_type,
                all_images, "Behavioral Agent", rag
            )
            st.markdown(resp_behavioral)
            all_retrieved['behavioral'] = retrieved_behavioral

        with st.spinner("Generating encouragement..."):
            st.subheader(" Strength & Motivation")
            resp_motivational, retrieved_motivational = safe_run_with_rag(
                motivational, prompt_motivational_template, user_input, issue_type,
                all_images, "Motivational Agent", rag
            )
            st.markdown(resp_motivational)
            all_retrieved['motivational'] = retrieved_motivational

        if st.session_state.enable_rag and rag and any(all_retrieved.values()):
            with st.expander(" Reference Sources (RAG Results)"):
                for agent_name, items in all_retrieved.items():
                    if items:
                        st.markdown(f"**{agent_name} Agent References:**")
                        for item in items:
                            st.markdown(f"- **{item['title']}** (Source: {item['source']}, Score: {item['score']:.2f})")
                            st.caption(f"  Preview: {item['content'][:150]}...")
                        st.markdown("---")

        combined_response = f"""Emotional Support:{resp_empathy}
Cognitive Restructuring:{resp_cognitive}
Behavioral Support:{resp_behavioral}
Motivational Support:{resp_motivational}"""

        history_entry = {
            "input": user_input,
            "response": combined_response,
            "files": [f.name for f in uploaded_files],
            "timestamp": datetime.now().isoformat(),
            "issue_type": issue_type,
            "rag_enabled": st.session_state.enable_rag
        }
        st.session_state.history.append(history_entry)

        save_history()