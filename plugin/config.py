# config.py
import json
import traceback
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

# Config class to manage configuration
class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        # General configuration
        helper.copy("ai_model_type")  # AI model type: openai, google, local, none
        # OpenAI configuration
        helper.copy("openai.api_key")
        helper.copy("openai.api_endpoint")
        helper.copy("openai.model")
        helper.copy("openai.max_tokens")
        helper.copy("openai.temperature")
        # Local LLM configuration
        helper.copy("local_llm.model_path")
        helper.copy("local_llm.api_endpoint")
        # Google Gemini configuration
        helper.copy("google.api_key")
        helper.copy("google.api_endpoint")
        helper.copy("google.model")
        helper.copy("google.max_tokens")
        helper.copy("google.temperature")
        # Discourse configuration
        helper.copy("discourse_api_key")
        helper.copy("discourse_api_username")
        helper.copy("discourse_base_url")
        helper.copy("unsorted_category_id")
        helper.copy("matrix_to_discourse_topic")

        # Command triggers
        helper.copy("search_trigger")
        helper.copy("post_trigger")
        helper.copy("help_trigger")
        helper.copy("url_post_trigger")
        helper.copy("target_audience")

        # Handle URL patterns and blacklist separately
        if "url_patterns" in helper.base:
            self["url_patterns"] = list(helper.base["url_patterns"])
        else:
            self["url_patterns"] = ["https?://.*"]  # Default to match all URLs

        if "url_blacklist" in helper.base:
            self["url_blacklist"] = list(helper.base["url_blacklist"])
        else:
            self["url_blacklist"] = []