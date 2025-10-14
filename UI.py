# UI.py  —— 含余额不足友好提示
import streamlit as st
from agent import build_agents, ModelChoice
from utils import process_images, logger
from agno.media import Image as AgnoImage
from agno.exceptions import ModelProviderError
from pathlib import Path
import tempfile
import os

st.set_page_config(page_title="Emotional Recovery AI Assistant", page_icon="😀", layout="wide")

with st.sidebar:
    st.header("⚙️ Model & API Configuration")

    # 1. 选择模型
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "gemini"
    model_choice: ModelChoice = st.selectbox(
        "Choose your model",
        options=["gemini", "openai", "claude", "deepseek"],
        index=["gemini", "openai", "claude", "deepseek"].index(st.session_state.model_choice),
        help="Select the model you want to use, then enter the corresponding API key below."
    )
    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice

    # 2. 输入对应 API Key
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    api_key = st.text_input(
        f"Enter your {model_choice.upper()} API Key",
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
        st.warning("Please enter your API key to proceed")
        links = {
            "gemini": "https://makersuite.google.com/app/apikey",
            "openai": "https://platform.openai.com/api-keys",
            "claude": "https://console.anthropic.com/settings/keys",
            "deepseek": "https://platform.deepseek.com/api-keys"
        }
        st.markdown(f"""
        To get your API key：  
        please visit: [{model_choice.upper()} Official]({links[model_choice]})
        """)

st.title("Emotional Recovery AI Assistant")
st.markdown("""### Your personal emotional recovery AI assistant is here to help you!
Share your feelings and chat screenshots, and I will offer you customized suggestions.""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Share Your Feelings")
    user_input = st.text_area("How are you feeling? What happened?", height=150,
                              placeholder="Tell us your story...")
with col2:
    st.subheader("Upload Chat Screenshots")
    uploaded_files = st.file_uploader("Upload screenshots (optional)", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

if st.button("Get Recovery Plan", type="primary"):
    if not st.session_state.api_key:
        st.warning("Please enter your API key in the sidebar first!")
        st.stop()

    agents = build_agents(st.session_state.api_key, st.session_state.model_choice)
    if not all(agents):
        st.error("Failed to initialize agents. Check API key.")
        st.stop()
    therapist, closure, routine, brutal = agents

    if not user_input and not uploaded_files:
        st.warning("Please share your feelings or upload screenshots to get help.")
        st.stop()

    all_images = process_images(uploaded_files) if uploaded_files else []
    st.header("Your Personalized Recovery Plan")

    def safe_run(agent, prompt, images):
        try:
            return agent.run(message=prompt, images=images).content
        except ModelProviderError as e:
            if "Insufficient Balance" in str(e) or "quota" in str(e).lower():
                st.error(
                    f"💰 **{st.session_state.model_choice.upper()} 账户余额不足！** \n\n"
                    f"请前往官方控制台充值，或换用其它模型后再试。"
                )
            else:
                st.error(f"模型调用失败：{e}")
            st.stop()
        except Exception as e:
            logger.error(f"Agent run error: {e}")
            st.error("生成内容时出现异常，请稍后再试。")
            st.stop()

    with st.spinner("Getting empathetic support..."):
        prompt_t = (f"User's message: {user_input}\nProvide compassionate response with validation, comfort, "
                    f"relatable experiences and encouragement.")
        st.subheader("Emotional Support")
        st.markdown(safe_run(therapist, prompt_t, all_images))

    with st.spinner("Crafting closure messages..."):
        prompt_c = (f"User's feelings: {user_input}\nProvide unsent message templates, emotional release "
                    f"exercises, closure rituals, moving forward strategies.")
        st.subheader("Finding Closure")
        st.markdown(safe_run(closure, prompt_c, all_images))

    with st.spinner("Creating your recovery plan..."):
        prompt_r = (f"Current state: {user_input}\nDesign 7-day recovery plan with daily activities, self-care "
                    f"routines, social media guidelines, playlists.")
        st.subheader("Your Recovery Plan")
        st.markdown(safe_run(routine, prompt_r, all_images))

    with st.spinner("Getting honest perspective..."):
        prompt_b = (f"Situation: {user_input}\nProvide objective analysis, growth opportunities, future outlook, "
                    f"actionable steps.")
        st.subheader("Honest Perspective")
        st.markdown(safe_run(brutal, prompt_b, all_images))

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""<div style='text-align:center'><p>由Data Mining小组制作</p>
<p>我们衷心的希望您在这里修复情感</p></div>""", unsafe_allow_html=True)