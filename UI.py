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

# --- 1. Page Configuration ---
# 'wide' layout uses the full page width
st.set_page_config(page_title="Emotional Recovery AI Assistant", page_icon="😀", layout="wide")

# --- Initialize Session State ---
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "gemini"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "history" not in st.session_state:
    st.session_state.history = []


# --- Helper: 问题类型分类 ---
def classify_issue_type(text: str) -> str:
    """智能识别用户情感问题类型"""
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
              " workplace ", "boss ", " colleague ", "performance ", " overtime ", "layoffs"]):
        return "workplace stress"
    elif any(kw in text_lower for kw in
             ["焦虑", "抑郁", "压力", "失眠", "情绪", "心理", "难受", "anxiety", "depressed", "stress",
              "insomnia ", "emotion ", " psychology ", "discomfort"]):
        return "mental health"
    elif any(kw in text_lower for kw in
             ["家人", "家庭", "父母", "亲戚", "沟通", "代沟", "family", "parents", "relatives ",
              "communication ", "generation gap"]):
        return "family issues"
    elif any(k in text_lower for k in
             ["钱", "经济", "贫穷", "债务", "买不起", "Money ", " economy ", "poverty ", " debt ", "unaffordable"]):
        return "financial stress"
    elif any(k in text_lower for k in
             ["考试", "挂科", "学习", "学业", "论文", "毕业", "gpa", "成绩", "Exam ", " Fail ", "Study ",
              " academic performance ", "thesis ", " graduation ", "gpa", "grade"]):
        return "academic anxiety"
    else:
        return "general emotional distress"


def save_history():
    """将对话历史保存到文件"""
    try:
        Path("conversation_history.json").write_text(
            json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")


# --- 2. Left Sidebar (Chat History) ---
# This is now docked to the far left edge
with st.sidebar:
    st.header("📜 Chat History")
    if not st.session_state.history:
        st.info("Your past submissions will appear here.")
    else:
        # Display history items
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.container(border=True):
                st.markdown(f"**{len(st.session_state.history) - i}:** {item['input'][:40]}...")
                if item['files']:
                    st.caption(f"📄 {len(item['files'])} files")
                    st.caption(f"🏷️ {item.get('issue_type', 'general')}")

    st.markdown("---")
    st.markdown("""<div style='text-align:center'><p>由Data Mining小组制作</p>
    <p>我们衷心的希望您在这里修复情感</p></div>""", unsafe_allow_html=True)

# --- 3. Main Page Layout (Center + Right) ---
# 70% for main chat, 30% for config
center_col, right_col = st.columns([0.7, 0.3])

# --- 4. Right Column (Configuration) ---
with right_col:
    with st.container(border=True):
        st.header("⚙️ Configuration")

        # 1. 选择模型
        model_choice: ModelChoice = st.selectbox(
            "Choose your model",
            options=["gemini", "openai", "claude", "deepseek"],
            index=["gemini", "openai", "claude", "deepseek"].index(st.session_state.model_choice),
            help="Select the model you want to use, then enter the corresponding API key below."
        )
        if model_choice != st.session_state.model_choice:
            st.session_state.model_choice = model_choice

        # 2. 输入对应 API Key
        api_key = st.text_input(
            f"Enter {model_choice.upper()} API Key",
            value=st.session_state.api_key,
            type="password",
            help=f"Get your key from the official website of {model_choice.upper()}"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key

        # 3. 快速指引
        if api_key:
            st.success("API Key provided! ✅")
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

# --- 5. Center Column (Main App "Chat Box") ---
with center_col:
    st.title("Emotional Recovery AI Assistant")
    st.markdown("""### Your personal emotional recovery AI assistant is here to help you!
    Share your feelings and images, and receive evidence-based support tailored to your situation.""")
    st.divider()

    # --- Input "Chat Box" Area ---
    with st.container(border=True):
        st.subheader("Share Your Feelings")
        user_input = st.text_area("How are you feeling? What happened?", height=150,
                                  placeholder="Tell us your story...", label_visibility="collapsed")

        # Place file uploader and submit button on the same row
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            uploaded_files = st.file_uploader("Upload Chat Screenshots (optional)",
                                              type=["jpg", "jpeg", "png"],
                                              accept_multiple_files=True)
        with col2:
            # Main submit button
            submit_button = st.button("Get Emotional Assistance", type="primary", use_container_width=True,
                                      help="Click to generate your recovery plan")

        # Preview images if they are uploaded
        if uploaded_files:
            with st.expander("View Uploaded Images"):
                for file in uploaded_files:
                    st.image(file, caption=file.name, width='stretch')

    # --- Logic and Output Area (displays below the chat box) ---
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
                # 1. 先轻度高斯模糊去噪
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                # 2. 大核自适应阈值
                binary = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 8)
                # 3. 2 倍放大
                h, w = binary.shape
                binary = cv2.resize(binary, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                file.seek(0)
                return pytesseract.image_to_string(binary, lang="chi_sim+eng")


            ocr_texts = []
            for file in uploaded_files:
                text = ocr_image(file)
                ocr_texts.append(f"【Image {file.name}】\n{text}")
            user_input = "\n\n".join(ocr_texts) + "\n\n" + (user_input or "")

            with st.expander("📄 OCR 原始结果（调试）"):
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

        st.divider()
        st.header("🌱 Your Personalized Recovery Plan")


        def safe_run(agent, prompt, images):
            try:
                return agent.run(input=prompt, images=images).content
            except ModelProviderError as e:
                if "Insufficient Balance" in str(e) or "quota" in str(e).lower():
                    st.error(
                        f"💰 **{st.session_state.model_choice.upper()} 账户余额不足！**\n\n"
                        f"请前往官方控制台充值，或换用其它模型后再试。"
                    )
                else:
                    st.error(f"模型调用失败 (ModelProviderError): {e}")
                logger.error(f"ModelProviderError: {e}")
                st.stop()
            except Exception as e:
                logger.error(f"Agent run error: {e}")
                st.error(f"生成内容时出现异常: {e}")
                st.stop()


        resp_empathy = resp_cognitive = resp_behavioral = resp_motivational = ""
        # (1) Empathy Agent
        with st.spinner("Analyzing your emotional state..."):
            prompt_empathy = f"""YOUR TASK - EMOTIONAL VALIDATION:

        User's Situation ({issue_type}): "{user_input}"

        MANDATORY STEPS:
        1. Quote or paraphrase a specific part of their message
        2. State their emotion explicitly: "I understand you're feeling [emotion]..."
        3. Validate WHY this emotion makes sense in THEIR context
        4. Share ONE brief relatable experience about {issue_type}
        5. End with personalized encouragement using THEIR words

        CRITICAL: Your response must reference their specific situation, not generic platitudes."""

            st.subheader("💖 Emotional Validation & Support")
            resp_empathy = safe_run(empathy, prompt_empathy, all_images)
            st.markdown(resp_empathy)

        # (2) Cognitive Restructuring Agent
        with st.spinner("Identifying thought patterns..."):
            prompt_cognitive = f"""YOUR TASK - COGNITIVE RESTRUCTURING:

        User's Challenge ({issue_type}): "{user_input}"

        REQUIRED APPROACH:
        1. Identify 1-2 specific thought distortions in THEIR story (quote their words)
        2. Explain how THEIR specific thinking pattern is unhelpful
        3. Offer 2 alternative perspectives tailored to {issue_type}
        4. Use Socratic questions referencing THEIR situation

        FORBIDDEN: Generic CBT theory without connection to their story."""

            st.subheader("🧠 Cognitive Restructuring")
            resp_cognitive = safe_run(cognitive, prompt_cognitive, all_images)
            st.markdown(resp_cognitive)

        # (3) Behavioral Support Agent
        with st.spinner("Creating action plan..."):
            prompt_behavioral = f"""YOUR TASK - ACTIONABLE PLAN:

        User's Context ({issue_type}): "{user_input}"

        CREATE A 7-DAY PLAN SPECIFIC TO THEIR SITUATION:
        Day 1-2: Immediate coping for THEIR specific stressors
        Day 3-4: Activities that address THEIR pain points
        Day 5-6: Social media boundaries for {issue_type}
        Day 7: Reflection on THEIR progress

        RULE: Every suggestion must connect to details in their story. No generic advice."""

            st.subheader("🎯 Practical Coping Strategies")
            resp_behavioral = safe_run(behavioral, prompt_behavioral, all_images)
            st.markdown(resp_behavioral)

        # (4) Motivational Agent
        with st.spinner("Generating encouragement..."):
            prompt_motivational = f"""YOUR TASK - PERSONALIZED MOTIVATION:

        User's Struggle ({issue_type}): "{user_input}"

        REQUIRED STRUCTURE:
        1. Reference THEIR past resilience (ask: what have they overcome?)
        2. Connect THEIR strength to THIS specific challenge
        3. Use THEIR words to show deep understanding
        4. Provide 3 encouraging next steps for THEIR situation

        ABSOLUTELY NO generic motivational quotes. Make it deeply personal."""

            st.subheader("💪 Strength & Motivation")
            resp_motivational = safe_run(motivational, prompt_motivational, all_images)
            st.markdown(resp_motivational)

        # 在所有Agent完成后，保存历史记录
        combined_response = f"""情感支持:{resp_empathy}
                                认知重构:{resp_cognitive}
                                行为支持:{resp_behavioral}
                                动机强化:{resp_motivational}"""

        # 添加到session state
        history_entry = {
            "input": user_input,
            "response": combined_response,
            "files": [f.name for f in uploaded_files],
            "timestamp": datetime.now().isoformat(),
            "issue_type": issue_type
        }
        st.session_state.history.append(history_entry)

        # 保存到文件
        save_history()
