import logging

log = logging.getLogger()
logging.getLogger("google_adk.google.adk.models.google_llm").setLevel(logging.WARNING)
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)
logging.getLogger("google_adk.google.adk.plugins.plugin_manager").setLevel(logging.WARNING)
