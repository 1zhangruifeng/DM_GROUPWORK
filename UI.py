# UI.py
import streamlit as st
from agent import build_agents, ModelChoice
from utils import process_images, logger
from agno.media import Image as AgnoImage
from agno.exceptions import ModelProviderError
from pathlib import Path
import tempfile
import os

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
                "gemini": "https.makersuite.google.com/app/apikey",
                "openai": "https.platform.openai.com/api-keys",
                "claude": "https.console.anthropic.com/settings/keys",
                "deepseek": "https.platform.deepseek.com/api-keys"
            }
            st.markdown(f"""
            To get your API key:
            please visit: [{model_choice.upper()} Official]({links[model_choice]})
            """)

# --- 5. Center Column (Main App "Chat Box") ---
with center_col:
    st.title("Emotional Recovery AI Assistant")
    st.markdown("""### Your personal emotional recovery AI assistant is here to help you!
    Share your feelings and chat screenshots, and I will offer you customized suggestions.""")

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
                    st.image(file, caption=file.name, use_container_width=True)

    # --- Logic and Output Area (displays below the chat box) ---
    if submit_button:
        if not st.session_state.api_key:
            st.warning("Please enter your API key in the configuration panel on the right!")
            st.stop()

        if not user_input and not uploaded_files:
            st.warning("Please share your feelings or upload screenshots to get help.")
            st.stop()

        try:
            agents = build_agents(st.session_state.api_key, st.session_state.model_choice)
            if not all(agents):
                st.error("Failed to initialize agents. Check API key and model choice.")
                st.stop()
            therapist, closure, routine, brutal = agents
        except Exception as e:
            st.error(f"Failed to build agents: {e}. Please check your API key.")
            logger.error(f"Agent build error: {e}")
            st.stop()

        all_images = process_images(uploaded_files) if uploaded_files else []

        # Add to history (this will be in session state)
        history_entry = {"input": user_input, "files": [f.name for f in uploaded_files]}
        st.session_state.history.append(history_entry)

        st.divider()
        st.header("Your Personalized Recovery Plan")


        def safe_run(agent, prompt, images):
            try:
                return agent.run(message=prompt, images=images).content
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


        with st.spinner("Getting empathetic support..."):
            prompt_t = (f"User's message: {user_input}\nProvide compassionate response with validation, comfort, "
                        f"relatable experiences and encouragement.")
            st.subheader("💖 Emotional Support")
            st.markdown(safe_run(therapist, prompt_t, all_images))

        with st.spinner("Crafting closure messages..."):
            prompt_c = (f"User's feelings: {user_input}\nProvide unsent message templates, emotional release "
                        f"exercises, closure rituals, moving forward strategies.")
            st.subheader("✨ Finding Closure")
            st.markdown(safe_run(closure, prompt_c, all_images))

        with st.spinner("Creating your recovery plan..."):
            prompt_r = (f"Current state: {user_input}\nDesign 7-day recovery plan with daily activities, self-care "
                        f"routines, social media guidelines, playlists.")
            st.subheader("🗓️ Your Recovery Plan")
            st.markdown(safe_run(routine, prompt_r, all_images))

        with st.spinner("Getting honest perspective..."):
            prompt_b = (f"Situation: {user_input}\nProvide objective analysis, growth opportunities, future outlook, "
                        f"actionable steps.")
            st.subheader("🔍 Honest Perspective")
            st.markdown(safe_run(brutal, prompt_b, all_images))
