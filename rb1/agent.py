from google.adk.agents import Agent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.google_llm import Gemini as LLM
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=3,
    http_status_codes=[429, 500, 503, 504],
)

root_agent = Agent(
    model=LLM(
        model="gemini-2.5-flash",
        retry_options=retry_config),
    name="rb1_root",
    description="",
    instruction="""
    
    """,
    tools=[],
    output_key="user_task"
)

app = App(
    name="agents",
    root_agent=root_agent,
    events_compaction_config=None
)
