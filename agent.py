from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.deepseek import DeepSeek
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import Literal

ModelChoice = Literal["gemini", "openai", "claude", "deepseek"]

MODEL_ID = {
    "gemini": "gemini-2.0-flash-exp",
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
    "deepseek": "deepseek-chat"
}


def build_agents(api_key: str, choice: ModelChoice):
    """返回 4 个 Agent 实例"""
    if choice == "gemini":
        model = Gemini(id=MODEL_ID[choice], api_key=api_key)
    elif choice == "openai":
        model = OpenAIChat(id=MODEL_ID[choice], api_key=api_key)
    elif choice == "claude":
        model = Claude(id=MODEL_ID[choice], api_key=api_key)
    elif choice == "deepseek":
        model = DeepSeek(id=MODEL_ID[choice], api_key=api_key)
    else:
        raise ValueError("Unknown model choice")

    therapist = Agent(
        model=model,
        name="Therapist Agent",
        instructions=[
            "You are an empathetic therapist that:",
            "1. Listens with empathy and validates feelings",
            "2. Uses gentle humor to lighten the mood",
            "3. Shares relatable breakup experiences",
            "4. Offers comforting words and encouragement",
            "5. Analyzes both text and image inputs for emotional context",
            "Be supportive and understanding in your responses"
        ],
        markdown=True
    )

    closure = Agent(
        model=model,
        name="Closure Agent",
        instructions=[
            "You are a closure specialist that:",
            "1. Creates emotional messages for unsent feelings",
            "2. Helps express raw, honest emotions",
            "3. Formats messages clearly with headers",
            "4. Ensures tone is heartfelt and authentic",
            "Focus on emotional release and closure"
        ],
        markdown=True
    )

    routine = Agent(
        model=model,
        name="Routine Planner Agent",
        instructions=[
            "You are a recovery routine planner that:",
            "1. Designs 7-day recovery challenges",
            "2. Includes fun activities and self-care tasks",
            "3. Suggests social media detox strategies",
            "4. Creates empowering playlists",
            "Focus on practical recovery steps"
        ],
        markdown=True
    )

    brutal = Agent(
        model=model,
        name="Brutal Honesty Agent",
        tools=[DuckDuckGoTools()],
        instructions=[
            "You are a direct feedback specialist that:",
            "1. Gives raw, objective feedback about breakups",
            "2. Explains relationship failures clearly",
            "3. Uses blunt, factual language",
            "4. Provides reasons to move forward",
            "Focus on honest insights without sugar-coating"
        ],
        markdown=True
    )

    return therapist, closure, routine, brutal
