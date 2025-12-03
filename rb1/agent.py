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
    plugins=[ReflectAndRetryToolPlugin(max_retries=1)],
    events_compaction_config=EventsCompactionConfig(
        summarizer=LlmEventSummarizer(LLM(model="gemini-2.5-flash")),
        compaction_interval=3,
        overlap_size=0,
    ),
)
