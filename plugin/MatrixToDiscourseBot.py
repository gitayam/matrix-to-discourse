# MatrixToDiscourseBot.py is the main file for the MatrixToDiscourseBot plugin.
import asyncio
import json
import re
import traceback
import aiohttp
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Type, Dict, List, Optional
from urllib.parse import urlparse
import random
import os
from mautrix.client import Client
from mautrix.types import (
    Format,
    TextMessageEventContent,
    EventType,
    RelationType,
    MessageType,
    PaginationDirection,
)
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper
from mautrix.util.async_db import UpgradeTable
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# BeautifulSoup4 since maubot image uses alpine linux
from bs4 import BeautifulSoup

# Import our database module
from .db import upgrade_table, MessageMappingDatabase

# Configure logging
logger = logging.getLogger(__name__)

# Config class to manage configuration
class Config(BaseProxyConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compile regex patterns once during initialization for performance
        self._compiled_blacklist_patterns = None
        self._compiled_whitelist_patterns = None
        self._patterns_compiled = False

    def _compile_patterns(self):
        """Compile regex patterns for URL filtering to improve performance."""
        if self._patterns_compiled:
            return
            
        try:
            blacklist = self.get("url_blacklist", [])
            whitelist = self.get("url_patterns", ["https?://.*"])
            
            self._compiled_blacklist_patterns = []
            self._compiled_whitelist_patterns = []
            
            for pattern in blacklist:
                try:
                    self._compiled_blacklist_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid blacklist regex pattern '{pattern}': {e}")
            
            for pattern in whitelist:
                try:
                    self._compiled_whitelist_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid whitelist regex pattern '{pattern}': {e}")
            
            self._patterns_compiled = True
            logger.debug(f"Compiled {len(self._compiled_blacklist_patterns)} blacklist and {len(self._compiled_whitelist_patterns)} whitelist patterns")
        except Exception as e:
            logger.error(f"Error compiling URL patterns: {e}")
            # Fallback to empty lists
            self._compiled_blacklist_patterns = []
            self._compiled_whitelist_patterns = []
            self._patterns_compiled = True

    def do_update(self, helper: ConfigUpdateHelper) -> None:
        # AI model type
        helper.copy("ai_model_type")  # AI model type: openai, google, local, none
        
        # OpenAI configuration
        helper.copy("openai.api_key")
        helper.copy("openai.api_endpoint")
        helper.copy("openai.model")
        helper.copy("openai.max_tokens")
        helper.copy("openai.temperature")
        
        # Local LLM configuration
        helper.copy("local_llm.connection_type")
        helper.copy("local_llm.api_endpoint")
        helper.copy("local_llm.api_format")
        helper.copy("local_llm.auth_method")
        helper.copy("local_llm.api_key")
        helper.copy("local_llm.model_path")
        helper.copy("local_llm.model")
        helper.copy("local_llm.max_tokens")
        helper.copy("local_llm.temperature")
        helper.copy("local_llm.top_p")
        
        # Google Gemini configuration
        helper.copy("google.api_key")
        helper.copy("google.api_endpoint")
        helper.copy("google.model")
        helper.copy("google.max_tokens")
        helper.copy("google.temperature")
        
        # Discourse configuration
        helper.copy("discourse_api_key")  # API key for discourse
        helper.copy("discourse_api_username")  # Username for discourse
        helper.copy("discourse_base_url")  # Base URL for discourse
        helper.copy("unsorted_category_id")  # Default category ID
        helper.copy("matrix_to_discourse_topic")  # Room-to-topic ID map

        # Command triggers
        helper.copy("search_trigger") # trigger for the search command
        helper.copy("post_trigger") # trigger for the post command
        helper.copy("help_trigger") # trigger for the help command
        helper.copy("url_listening") # trigger for the url listening
        helper.copy("url_post_trigger") # trigger for the url post command
        helper.copy("target_audience") # target audience context for ai generation
        helper.copy("summary_length_in_characters") # length of the summary preview
        
        # Admin configuration
        helper.copy("admins")  # List of admin user IDs
        
        # MediaWiki configuration
        helper.copy("mediawiki_base_url")  # Base URL for the community MediaWiki wiki
        helper.copy("mediawiki_search_trigger")  # Command trigger for MediaWiki wiki search
        helper.copy("mediawiki.api_endpoint")
        helper.copy("mediawiki.username")
        helper.copy("mediawiki.password")
        
        # URL patterns and blacklist
        helper.copy("url_patterns")
        helper.copy("url_blacklist")
        
        # Reset compiled patterns when config is updated
        self._patterns_compiled = False

    def should_process_url(self, url: str) -> bool:
        """
        Check if a URL should be processed based on patterns and blacklist.
        Uses compiled regex patterns for better performance.

        Args:
        url (str): The URL to check
        
        Returns:
            bool: True if the URL should be processed, False otherwise
        """
        if not url:
            return False
            
        # Ensure patterns are compiled
        self._compile_patterns()
        
        # Input sanitization - basic URL validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logger.debug(f"Invalid URL format: {url}")
                return False
        except Exception as e:
            logger.warning(f"Error parsing URL {url}: {e}")
            return False
        
        # First check blacklist using compiled patterns
        for pattern in self._compiled_blacklist_patterns:
            try:
                if pattern.search(url):
                    logger.debug(f"URL {url} matches blacklist pattern {pattern.pattern}")
                    return False
            except Exception as e:
                logger.warning(f"Error matching blacklist pattern: {e}")
                continue
        
        # To prevent infinite recursion, exclude URLs pointing to the bot's own mediawiki or discourse instances if configured
        # Exclude URLs pointing to the bot's own mediawiki or discourse instances if configured
        if 'mediawiki_base_url' in self and self['mediawiki_base_url']:
            if self['mediawiki_base_url'] in url:
                logger.debug(f"URL {url} matches mediawiki_base_url {self['mediawiki_base_url']}")
                return False

        if 'discourse_base_url' in self and self['discourse_base_url']:
            if self['discourse_base_url'] in url:
                logger.debug(f"URL {url} matches discourse_base_url {self['discourse_base_url']}")
                return False

            # Extract base domain from discourse_base_url
            # many services such as notepads, and other links are typical in the communtiy chats but create redudant posts in discourse
            try:
                parsed_url = urlparse(self['discourse_base_url'])
                base_domain = parsed_url.netloc #netloc is the network location of the URL

                # Add base domain and all subdomains to the blacklist
                if base_domain in url:
                    logger.debug(f"URL {url} matches base domain {base_domain} of discourse_base_url")
                    return False
            except Exception as e:
                logger.warning(f"Error parsing discourse_base_url: {e}")

        # Then check whitelist patterns using compiled patterns
        for pattern in self._compiled_whitelist_patterns:
            try:
                if pattern.search(url):
                    logger.debug(f"URL {url} matches whitelist pattern {pattern.pattern}")
                    return True
            except Exception as e:
                logger.warning(f"Error matching whitelist pattern: {e}")
                continue
    
        # If no patterns match, don't process the URL
        return False

    def safe_config_get(self, key: str, default=None):
        """
        Safely get a value from the config, handling RecursiveDict properly.
        
        Args:
            key (str): The config key to get
            default: The default value to return if the key is not found
            
        Returns:
            The value from the config, or the default value if not found
        """
        try:
            # Check if this is a RecursiveDict (mautrix config type)
            if 'RecursiveDict' in str(type(self)):
                # RecursiveDict requires the default value as a positional argument
                try:
                    return self.get(key, default)
                except TypeError:
                    # If the signature is different, try with positional args
                    try:
                        if key in self:
                            return self[key]
                        else:
                            return default
                    except Exception:
                        return default
            elif hasattr(self, 'get'):
                # Standard dict-like get method
                return self.get(key, default)
            elif isinstance(self, dict) and key in self:
                return self[key]
            else:
                return default
        except Exception as e:
            logger.warning(f"Error getting config value for key {key}: {e}")
            return default

    def validate_config(self):
        """Validate critical configuration parameters with format validation."""
        critical_params = [
            ("discourse_base_url", "Discourse base URL", lambda x: x and x.startswith("http")),
            ("discourse_api_key", "Discourse API key", lambda x: x and len(x.strip()) > 0),
            ("discourse_api_username", "Discourse API username", lambda x: x and len(x.strip()) > 0),
        ]
        
        missing = []
        for param, desc, validator in critical_params:
            value = self.safe_config_get(param)
            if not value or not validator(value):
                missing.append(desc)
        
        if missing:
            logger.error(f"Invalid configuration: {', '.join(missing)}")
            return False
        
        # Validate AI model specific config
        ai_model_type = self.safe_config_get("ai_model_type", "none")
        if ai_model_type != "none":
            if not self._validate_ai_config(ai_model_type):
                return False
        
        return True

    def _validate_ai_config(self, model_type: str) -> bool:
        """Validate AI model specific configuration."""
        required_fields = {
            "openai": ["api_key", "api_endpoint", "model"],
            "google": ["api_key", "api_endpoint", "model"],
            "local": ["api_endpoint", "model"]
        }
        
        if model_type not in required_fields:
            logger.error(f"Unknown AI model type: {model_type}")
            return False
        
        missing = []
        for field in required_fields[model_type]:
            config_key = f"{model_type}.{field}"
            value = self.safe_config_get(config_key)
            if not value or (isinstance(value, str) and len(value.strip()) == 0):
                missing.append(field)
        
        if missing:
            logger.error(f"Missing required fields for {model_type}: {', '.join(missing)}")
            return False
        
        # Additional validation for specific fields
        if model_type in ["openai", "google", "local"]:
            endpoint = self.safe_config_get(f"{model_type}.api_endpoint")
            if endpoint and not endpoint.startswith("http"):
                logger.error(f"Invalid API endpoint for {model_type}: must start with http")
                return False
        
        return True

# AIIntegration class
class AIIntegration:
    """
    A class to handle AI integration for the plugin.
    Includes methods for generating titles, summaries, and tags.
    """
    def __init__(self, config, log):
        self.config = config
        self.logger = log  # Change logger to self.logger
        self.target_audience = self.safe_config_get("target_audience", "community members")
        # Initialize Discourse API
        self.discourse_api = DiscourseAPI(self.config, log)  # Pass log directly

    def safe_config_get(self, key: str, default=None):
        """
        Safely get a value from the config, handling RecursiveDict properly.
        """
        try:
            # Check if this is a RecursiveDict (mautrix config type)
            if 'RecursiveDict' in str(type(self.config)):
                # RecursiveDict requires the default value as a positional argument
                try:
                    return self.config.get(key, default)
                except TypeError:
                    # If the signature is different, try with positional args
                    try:
                        if key in self.config:
                            return self.config[key]
                        else:
                            return default
                    except Exception:
                        return default
            elif hasattr(self.config, 'get'):
                # Standard dict-like get method
                return self.config.get(key, default)
            elif isinstance(self.config, dict) and key in self.config:
                return self.config[key]
            else:
                return default
        except Exception as e:
            self.logger.warning(f"Error getting config value for key {key}: {e}")
            return default

    def clean_markdown_from_title(self, title: str) -> str:
        """Remove markdown formatting from a title."""
        if not title:
            return title
            
        # Remove markdown headers (# Header)
        title = re.sub(r'^#+\s+', '', title)
        
        # Remove markdown bold/italic (**bold**, *italic*)
        title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
        title = re.sub(r'\*(.*?)\*', r'\1', title)
        
        # Remove markdown links ([text](url))
        title = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', title)
        
        # Remove backticks (`code`)
        title = re.sub(r'`(.*?)`', r'\1', title)
        
        return title.strip()

    async def generate_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        ai_model_type = self.safe_config_get("ai_model_type", "none")

        if ai_model_type == "openai":
            return await self.generate_openai_title(message_body, use_links_prompt)
        elif ai_model_type == "local":
            return await self.generate_local_title(message_body, use_links_prompt)
        elif ai_model_type == "google":
            return await self.generate_google_title(message_body, use_links_prompt)
        else:
            self.logger.error(f"Unknown AI model type: {ai_model_type}")
            return None
    #generate_links_title
    async def generate_links_title(self, message_body: str) -> Optional[str]:
        """Create a title focusing on linked content."""
        prompt = f"Create a concise and relevant title for the following content, focusing on the key message and linked content:\n\n{message_body}"
        ai_model_type = self.safe_config_get("ai_model_type", "none")

        if ai_model_type == "openai":
            return await self.generate_openai_title(prompt, True)
        elif ai_model_type == "local":
            return await self.generate_local_title(prompt, True)
        elif ai_model_type == "google":
            return await self.generate_google_title(prompt, True)
        else:
            self.logger.error(f"Unknown AI model type: {ai_model_type}")
            return None

    async def summarize_content(self, content: str, user_message: str = "") -> Optional[str]:
        # If this is a prompt for key points, exactly 5 key points, or a specific summary format, use it directly
        if "bullet point format" in content or "concise summary" in content or "maximum 300 words" in content or "exactly 5 key points" in content:
            prompt = content
        else:
            # Otherwise, use the standard summary prompt
            prompt = f"""Summarize the following content, including any user-provided context or quotes. If there isn't enough information, please just say 'Not Enough Information to Summarize':\n\nContent:\n{content}\n\nUser Message:\n{user_message}"""
        
        ai_model_type = self.safe_config_get("ai_model_type", "none")
        if ai_model_type == "openai":
            return await self.summarize_with_openai(prompt)
        elif ai_model_type == "local":
            return await self.summarize_with_local_llm(prompt)
        elif ai_model_type == "google":
            return await self.summarize_with_google(prompt)
        else:
            logger.error(f"Unknown AI model type: {ai_model_type}")
            return None
    #generate_openai_tags
    TAG_PROMPT = """Analyze the following content and suggest 2-4 relevant tags {user_message}.  
        NOW Choose from or be inspired by these existing tags: {tag_list}
        If none of the existing tags fit well, suggest new related tags.
        The tags should be lowercase, use hyphens instead of spaces, and be concise.       

        Return only the tags as a comma-separated list, no explanation needed."""
    # user_message is the message body of the post used for context for tag generation
    async def generate_tags(self, user_message: str = "") -> Optional[List[str]]:
        try:
            if not self.discourse_api:
                self.logger.error("Discourse API is not initialized.")
                return ["posted-link"]  # Return default tag instead of None

            # Get existing tags from Discourse for context
            all_tags = await self.discourse_api.get_all_tags()
            if all_tags is None:
                self.logger.error("Failed to fetch tags from Discourse API.")
                return ["posted-link"]  # Return default tag

            # Log the type and content of all_tags for debugging
            self.logger.debug(f"Type of all_tags: {type(all_tags)}")
            self.logger.debug(f"Content of all_tags: {all_tags}")

            # Check if all_tags is a list and contains dictionaries
            if not isinstance(all_tags, list) or not all(isinstance(tag, dict) for tag in all_tags):
                self.logger.error("Unexpected format for all_tags. Expected a list of dictionaries.")
                return ["posted-link"]  # Return default tag

            tag_names = [tag["name"] for tag in all_tags]
            tag_list = ", ".join(list(dict.fromkeys(tag_names)))

            # Create prompt with existing tags context accepts tag list and content , pass user message for context
            prompt = self.TAG_PROMPT.format(tag_list=tag_list, user_message=user_message)

            ai_model_type = self.safe_config_get("ai_model_type", "none")
            
            if ai_model_type == "openai":
                return await self._generate_tags_openai(prompt)
            elif ai_model_type == "local":
                return await self._generate_tags_local(prompt)
            elif ai_model_type == "google":
                return await self._generate_tags_google(prompt)
            else:
                self.logger.error(f"Unknown AI model type: {ai_model_type}")
                return ["posted-link"]  # Return default tag

        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating tags: {e}\n{tb}")
            return ["posted-link"]  # Return default tag instead of "fix-me"

    async def _generate_tags_openai(self, prompt: str) -> List[str]:
        """Generate tags using OpenAI API with improved error handling and timeouts."""
        try:
            api_key = self.config.get('openai.api_key', None)
            if not api_key:
                self.logger.error("OpenAI API key is not configured.")
                return ["posted-link"]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            data = {
                "model": self.config.get("openai.model", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(self.config.get("openai.max_tokens", 500), 1000),
                "temperature": self.config.get("openai.temperature", 0.7),
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.config.get("openai.api_endpoint"), headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding OpenAI response: {e}\nResponse text: {response_text}")
                        return ["posted-link"]

                    if response.status == 200:
                        if "choices" not in response_json or not response_json["choices"]:
                            self.logger.error("Invalid OpenAI response format: no choices")
                            return ["posted-link"]
                            
                        tags_text = response_json["choices"][0]["message"]["content"].strip()
                        return self._process_tag_response(tags_text)
                    else:
                        self.logger.error(f"OpenAI API error: {response.status} {response_json}")
                        return ["posted-link"]
        except asyncio.TimeoutError:
            self.logger.error("OpenAI API request timed out")
            return ["posted-link"]
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating tags with OpenAI: {e}\n{tb}")
            return ["posted-link"]

    async def _generate_tags_local(self, prompt: str) -> List[str]:
        """Generate tags using Local LLM API with improved error handling and timeouts."""
        try:
            headers = {"Content-Type": "application/json"}
            
            # Add authentication if configured
            auth_method = self.config.get("local_llm.auth_method", "")
            api_key = self.config.get("local_llm.api_key", "")
            
            if auth_method == "bearer" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif auth_method == "api_key" and api_key:
                headers["X-API-Key"] = api_key
            
            # Use OpenAI-compatible format
            data = {
                "model": self.config.get("local_llm.model", "gemma-2-2b-it"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(self.config.get("local_llm.max_tokens", 500), 1000),
                "temperature": self.config.get("local_llm.temperature", 0.7),
            }

            timeout = aiohttp.ClientTimeout(total=60)  # Longer timeout for local models
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.config.get("local_llm.api_endpoint"), headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Local LLM response: {e}\nResponse text: {response_text}")
                        return ["posted-link"]

                    if response.status == 200:
                        # Handle OpenAI-compatible format
                        if "choices" in response_json and len(response_json["choices"]) > 0:
                            if "message" in response_json["choices"][0]:
                                tags_text = response_json["choices"][0]["message"]["content"].strip()
                                return self._process_tag_response(tags_text)
                        return ["posted-link"]
                    else:
                        self.logger.error(f"Local LLM API error: {response.status} {response_json}")
                        return ["posted-link"]
        except asyncio.TimeoutError:
            self.logger.error("Local LLM API request timed out")
            return ["posted-link"]
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating tags with Local LLM: {e}\n{tb}")
            return ["posted-link"]

    async def _generate_tags_google(self, prompt: str) -> List[str]:
        """Generate tags using Google API with improved error handling and timeouts."""
        try:
            api_key = self.config.get('google.api_key', None)
            if not api_key:
                self.logger.error("Google API key is not configured.")
                return ["posted-link"]

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key  # Google uses x-goog-api-key header
            }
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": min(self.config.get("google.max_tokens", 500), 1000),
                    "temperature": self.config.get("google.temperature", 0.7),
                }
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.config.get("google.api_endpoint"), headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Google API response: {e}\nResponse text: {response_text}")
                        return ["posted-link"]

                    if response.status == 200:
                        if "candidates" not in response_json or not response_json["candidates"]:
                            self.logger.error("Invalid Google API response format: no candidates")
                            return ["posted-link"]
                            
                        candidate = response_json["candidates"][0]
                        if "content" not in candidate or "parts" not in candidate["content"]:
                            self.logger.error("Invalid Google API response format: missing content/parts")
                            return ["posted-link"]
                            
                        tags_text = candidate["content"]["parts"][0]["text"].strip()
                        return self._process_tag_response(tags_text)
                    else:
                        self.logger.error(f"Google API error: {response.status} {response_json}")
                        return ["posted-link"]
        except asyncio.TimeoutError:
            self.logger.error("Google API request timed out")
            return ["posted-link"]
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating tags with Google: {e}\n{tb}")
            return ["posted-link"]

    def _process_tag_response(self, tags_text: str) -> List[str]:
        """Process and validate tag response from AI models."""
        if not tags_text:
            return ["posted-link"]
            
        # Split by comma and clean up tags
        tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
        
        # Filter out invalid tags (only allow alphanumeric and hyphens)
        valid_tags = []
        for tag in tags:
            if tag and re.match(r'^[a-z0-9\-]+$', tag) and len(tag) <= 50:  # Max length check
                valid_tags.append(tag)
        
        # Always include posted-link tag
        if "posted-link" not in valid_tags:
            valid_tags.append("posted-link")
        
        # Limit to 5 tags maximum (including posted-link)
        valid_tags = valid_tags[:5]
        
        # Ensure we have at least one tag
        if not valid_tags:
            return ["posted-link"]
            
        return valid_tags

    generate_title_prompt = "Create a brief (3-10 word) attention-grabbing title for the {target_audience} for the following post on the community forum: {message_body}"
    generate_links_title_prompt = "Create a brief (3-10 word) attention-grabbing title for the following post on the community forum include the source and title of the linked content: {message_body}"
    # function to generate a title will pass either generate_title_prompt or generate_links_title_prompt based on the use_links_prompt flag
    # Implement the methods for each AI model
    # when called generate_title_prompt or generate_links_title_prompt will be used
    async def generate_openai_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        # Choose the appropriate prompt based on the use_links_prompt flag
        if use_links_prompt:
            # use the generate_links_title_prompt if the use_links_prompt flag is true
            prompt_template = "Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        else:
            # if not use_links_prompt, use the generate_title_prompt
            prompt_template = "Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        
        prompt = prompt_template.format(message_body=message_body, target_audience=self.target_audience)
        
        try:
            api_key = self.config.get('openai.api_key', None)
            if not api_key:
                self.logger.error("OpenAI API key is not configured.")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            data = {
                "model": self.config["openai.model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(self.config["openai.max_tokens"], 100),  # Limit to 100 tokens for title
                "temperature": self.config["openai.temperature"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["openai.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding OpenAI response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        title = response_json["choices"][0]["message"]["content"].strip()
                        
                        # Clean up the title - remove any "Title:" prefix or quotes that the model might add
                        title = title.replace("Title:", "").strip()
                        title = title.strip('"\'')
                        
                        # Clean any markdown formatting
                        title = self.clean_markdown_from_title(title)
                        
                        # Ensure the title is not too long for Discourse (max 255 chars)
                        if len(title) > 250:
                            title = title[:247] + "..."
                            
                        return title
                    else:
                        self.logger.error(f"OpenAI API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating title with OpenAI: {e}\n{tb}")
            return None

    async def summarize_with_openai(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary which is relevant to the {self.target_audience} of the following content:\n\n{content}"
        
        # Check if this is a specific format request
        if "bullet point format" in content or "maximum 300 words" in content or "exactly 5 key points" in content:
            prompt = content  # Use the full prompt as is
            
        # Limit content length to prevent context length errors
        if len(prompt) > 4000:
            prompt = prompt[:4000] + "..."
            
        try:
            api_key = self.config.get('openai.api_key', None)
            if not api_key:
                self.logger.error("OpenAI API key is not configured.")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            data = {
                "model": self.config["openai.model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config["openai.max_tokens"],
                "temperature": self.config["openai.temperature"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["openai.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding OpenAI response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        result = response_json["choices"][0]["message"]["content"].strip()
                        
                        # For key points, ensure proper formatting
                        if "bullet point format" in content or "exactly 5 key points" in content:
                            # Convert any non-bullet point format to bullet points
                            if not any(line.strip().startswith('-') for line in result.split('\n')):
                                # Split by numbers or newlines or periods followed by space
                                lines = re.split(r'\n|(?:\d+\.\s*)|(?:\.\s+)', result)
                                # Filter out empty lines and format as bullet points
                                formatted_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        # Add period if missing
                                        if not line.endswith('.'):
                                            line += '.'
                                        formatted_lines.append(f"- {line}")
                                result = '\n'.join(formatted_lines)
                        
                        return result
                    else:
                        self.logger.error(f"OpenAI API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error summarizing with OpenAI: {e}\n{tb}")
            return None

    async def generate_local_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        # Choose the appropriate prompt based on the use_links_prompt flag
        if use_links_prompt:
            prompt_template = "Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        else:
            prompt_template = "Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        
        prompt = prompt_template.format(message_body=message_body, target_audience=self.target_audience)
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config.get("local_llm.model", "gemma-2-2b-it"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(self.config.get("local_llm.max_tokens", 500), 100),  # Limit to 100 tokens for title
                "temperature": self.config.get("local_llm.temperature", 0.7),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["local_llm.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Local LLM response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        # Handle OpenAI-compatible format
                        if "choices" in response_json and len(response_json["choices"]) > 0:
                            if "message" in response_json["choices"][0]:
                                result = response_json["choices"][0]["message"]["content"].strip()
                            elif "text" in response_json["choices"][0]:
                                result = response_json["choices"][0]["text"].strip()
                        # Fallback for other formats
                        elif "text" in response_json:
                            result = response_json["text"].strip()
                        else:
                            self.logger.error(f"Unexpected response format from Local LLM: {response_json}")
                            return None
                        
                        # Clean up the title - remove any "Title:" prefix or quotes that the model might add
                        result = result.replace("Title:", "").strip()
                        result = result.strip('"\'')
                        
                        # Clean any markdown formatting
                        result = self.clean_markdown_from_title(result)
                        
                        # Ensure the title is not too long for Discourse (max 255 chars)
                        if len(result) > 250:
                            result = result[:247] + "..."
                            
                        return result
                    else:
                        self.logger.error(f"Local LLM API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating title with local LLM: {e}\n{tb}")
            return None

    async def summarize_with_local_llm(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary which is relevant to the {self.target_audience} of the following content:\n\n{content}"
        
        # Check if this is a specific format request
        if "bullet point format" in content or "maximum 300 words" in content or "exactly 5 key points" in content:
            prompt = content  # Use the full prompt as is
        
        # Limit content length to prevent context length errors
        if len(prompt) > 2500:
            prompt = prompt[:2500] + "..."
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config.get("local_llm.model", "gemma-2-2b-it"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(self.config.get("local_llm.max_tokens", 500), 1000),  # Limit to 1000 tokens
                "temperature": self.config.get("local_llm.temperature", 0.7),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["local_llm.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Local LLM response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        # Handle OpenAI-compatible format
                        if "choices" in response_json and len(response_json["choices"]) > 0:
                            if "message" in response_json["choices"][0]:
                                result = response_json["choices"][0]["message"]["content"].strip()
                                
                                # For key points, ensure proper formatting
                                if "bullet point format" in content or "exactly 5 key points" in content:
                                    # Convert any non-bullet point format to bullet points
                                    if not any(line.strip().startswith('-') for line in result.split('\n')):
                                        # Split by numbers or newlines or periods followed by space
                                        lines = re.split(r'\n|(?:\d+\.\s*)|(?:\.\s+)', result)
                                        # Filter out empty lines and format as bullet points
                                        formatted_lines = []
                                        for line in lines:
                                            line = line.strip()
                                            if line:
                                                # Add period if missing
                                                if not line.endswith('.'):
                                                    line += '.'
                                                formatted_lines.append(f"- {line}")
                                        result = '\n'.join(formatted_lines)
                                
                                return result
                            elif "text" in response_json["choices"][0]:
                                return response_json["choices"][0]["text"].strip()
                        # Fallback for other formats
                        elif "text" in response_json:
                            return response_json["text"].strip()
                        else:
                            self.logger.error(f"Unexpected response format from Local LLM: {response_json}")
                            return None
                    else:
                        self.logger.error(f"Local LLM API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error summarizing with local LLM: {e}\n{tb}")
            return None

    async def generate_google_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        # Choose the appropriate prompt based on the use_links_prompt flag
        if use_links_prompt:
            prompt_template = "Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        else:
            prompt_template = "Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        
        prompt = prompt_template.format(message_body=message_body, target_audience=self.target_audience)
        
        try:
            api_key = self.config.get('google.api_key', None)
            if not api_key:
                self.logger.error("Google API key is not configured.")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": self.config["google.model"],
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": min(self.config["google.max_tokens"], 100),  # Limit to 100 tokens for title
                    "temperature": self.config["google.temperature"],
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["google.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Google API response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        title = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        
                        # Clean up the title - remove any "Title:" prefix or quotes that the model might add
                        title = title.replace("Title:", "").strip()
                        title = title.strip('"\'')
                        
                        # Clean any markdown formatting
                        title = self.clean_markdown_from_title(title)
                        
                        # Ensure the title is not too long for Discourse (max 255 chars)
                        if len(title) > 250:
                            title = title[:247] + "..."
                            
                        return title
                    else:
                        self.logger.error(f"Google API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error generating title with Google: {e}\n{tb}")
            return None

    async def summarize_with_google(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary which is relevant to the {self.target_audience} of the following content:\n\n{content}"
        
        # Check if this is a specific format request
        if "bullet point format" in content or "maximum 300 words" in content or "exactly 5 key points" in content:
            prompt = content  # Use the full prompt as is
            
        # Limit content length to prevent context length errors
        if len(prompt) > 4000:
            prompt = prompt[:4000] + "..."
            
        try:
            api_key = self.config.get('google.api_key', None)
            if not api_key:
                self.logger.error("Google API key is not configured.")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": self.config["google.model"],
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self.config["google.max_tokens"],
                    "temperature": self.config["google.temperature"],
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["google.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Google API response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        result = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        
                        # For key points, ensure proper formatting
                        if "bullet point format" in content or "exactly 5 key points" in content:
                            # Convert any non-bullet point format to bullet points
                            if not any(line.strip().startswith('-') for line in result.split('\n')):
                                # Split by numbers or newlines or periods followed by space
                                lines = re.split(r'\n|(?:\d+\.\s*)|(?:\.\s+)', result)
                                # Filter out empty lines and format as bullet points
                                formatted_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        # Add period if missing
                                        if not line.endswith('.'):
                                            line += '.'
                                        formatted_lines.append(f"- {line}")
                                result = '\n'.join(formatted_lines)
                        
                        return result
                    else:
                        self.logger.error(f"Google API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error summarizing with Google: {e}\n{tb}")
            return None

    async def generate_title_for_bypassed_links(self, message_body: str) -> Optional[str]:
        """Generate a title for a message with bypassed links."""
        if not self.ai_integration:
            self.ai_integration = AIIntegration(self.config, logger)
        
        # Use a more direct prompt for title generation
        prompt = f"Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        
        title = await self.ai_integration.generate_title(prompt)
        
        # Clean up the title - remove any "Title:" prefix or quotes that the model might add
        if title:
            title = title.replace("Title:", "").strip()
            title = title.strip('"\'')
            
            # Clean any markdown formatting
            title = self.clean_markdown_from_title(title)
            
            # Ensure the title is not too long for Discourse (max 255 chars)
            if len(title) > 250:
                title = title[:247] + "..."
        
        return title

    async def api_call_with_retry(self, func, *args, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"API call failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    @command.new(name="list-mappings", require_subcommand=False)
    async def list_mappings_command(self, evt: MessageEvent) -> None:
        """Command to list the current message ID mappings."""
        # Only allow admins to use this command
        if not await self.is_admin(evt.sender):
            await evt.reply("You don't have permission to use this command.")
            return
        
        if not self.message_id_map:
            await evt.reply("No message ID mappings found.")
            return
        
        # Format the mappings for display
        mappings = []
        for matrix_id, discourse_id in list(self.message_id_map.items())[:10]:  # Limit to 10 entries
            mappings.append(f"Matrix ID: {matrix_id} -> Discourse Topic: {discourse_id}")
        
        total = len(self.message_id_map)
        shown = min(10, total)
        
        message = f"Message ID Mappings ({shown} of {total} shown):\n\n" + "\n".join(mappings)
        if total > 10:
            message += f"\n\n... and {total - 10} more."
        
        await evt.reply(message)

# DiscourseAPI class
class DiscourseAPI:
    def __init__(self, config, log):
        self.config = config
        self.logger = log  # Change logger to self.logger
        
        # Ensure we have the required configuration
        self.base_url = config.get("discourse_base_url", None)
        self.api_key = config.get("discourse_api_key", None)
        self.api_username = config.get("discourse_api_username", None)
        
        if not self.base_url or not self.api_key or not self.api_username:
            self.logger.error("Missing required Discourse configuration. Please check your config file.")
    
    async def create_post(self, title, raw, category_id, tags=None, topic_id=None):
        """
        Create a post on Discourse.
        
        Args:
            title (str): The title of the post. Not needed for replies.
            raw (str): The raw content of the post.
            category_id (int): The category ID. Not needed for replies.
            tags (list): List of tags. Not needed for replies.
            topic_id (int, optional): The topic ID to reply to. If provided, creates a reply instead of a new topic.
            
        Returns:
            tuple: (post_url, error)
        """
        if not self.base_url or not self.api_key or not self.api_username:
            return None, "Missing required Discourse configuration"
            
        # Ensure title is within Discourse's 255 character limit
        if title and len(title) > 250:
            title = title[:247] + "..."
            
        url = f"{self.base_url}/posts.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.api_key,
            "Api-Username": self.api_username,
        }
        
        # Prepare the payload
        payload = {
            "raw": raw,
        }
        
        if topic_id:
            # This is a reply to an existing topic
            payload["topic_id"] = topic_id
            self.logger.info(f"Creating reply to topic ID: {topic_id}")
        else:
            # This is a new topic
            payload["title"] = title
            payload["category"] = category_id
            if tags:
                payload["tags"] = tags
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_text = await response.text()
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                    return None, f"Error decoding response: {e}"
                
                if response.status == 200:
                    # Extract the post URL from the response
                    post_id = response_json.get("id")
                    topic_id = response_json.get("topic_id")
                    topic_slug = response_json.get("topic_slug")
                    
                    if post_id and topic_id and topic_slug:
                        post_url = f"{self.base_url}/t/{topic_slug}/{topic_id}"
                        return post_url, None
                    else:
                        return None, "Missing post information in response"
                else:
                    data = response_json.get("errors", response_json)
                    self.logger.error(f"Discourse API error: {response.status} {data}")
                    return None, f"API error: {data}"
    # Check for duplicate posts with discourse api
    async def check_for_duplicate(self, url: str, tag: str = "posted-link") -> bool:
        """
        Check if a URL has already been posted to Discourse.
        
        Args:
            url (str): The URL to check
            tag (str): The tag to search for (default: "posted-link")
            
        Returns:
            tuple: (bool, str) - (True if duplicate exists, URL of the duplicate post)
        """
        try:
            # Define headers here to avoid the "headers is not defined" error
            headers = {
                "Content-Type": "application/json",
                "Api-Key": self.api_key,
                "Api-Username": self.api_username,
            }
            
            # Normalize the URL to handle variations
            normalized_url = self.normalize_url(url)
            self.logger.debug(f"Checking for duplicates of normalized URL: {normalized_url}")
            
            # First, try a direct search for the URL
            params = {"q": normalized_url}
            self.logger.debug(f"Searching Discourse with direct URL: {params}")
            
            async with aiohttp.ClientSession() as session:
                search_url = f"{self.base_url}/search.json"
                async with session.get(search_url, params=params, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(f"Discourse API error: {response.status}")
                        return False, None
                    
                    response_text = await response.text()
                    response_json = json.loads(response_text)
                    
                    # Check if there are any topics in the response
                    if "topics" in response_json and response_json["topics"]:
                        for topic in response_json["topics"]:
                            # Get the topic content and check if it contains the URL
                            topic_id = topic.get("id")
                            # Check if the topic has the URL in its content
                            if "id" in topic and "slug" in topic:
                                topic_url = f"{self.base_url}/t/{topic['slug']}/{topic['id']}"
                                self.logger.info(f"Duplicate post found: {topic_url}")
                                return True, topic_url
            
            # If no direct match, try searching for posts with the tag
            params = {"q": f"tags:{tag}"}
            self.logger.debug(f"Searching Discourse with tag: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Discourse API error: {response.status}")
                        return False, None

                    response_json = await response.json()
                    
                    # Check each post for the normalized URL in its "raw" content
                    for post in response_json.get("posts", []):
                        raw_content = post.get("raw", "")
                        post_id = post.get("id")
                        topic_id = post.get("topic_id")
                        
                        # Check for various URL formats that might be in the content
                        url_variations = [
                            normalized_url,
                            normalized_url.replace("https://", "http://"),
                            normalized_url.replace("http://", "https://"),
                            normalized_url.replace("https://www.", "https://"),
                            normalized_url.replace("http://www.", "http://"),
                            normalized_url.replace("https://", "https://www."),
                            normalized_url.replace("http://", "http://www.")
                        ]
                        
                        for variation in url_variations:
                            if variation in raw_content:
                                if topic_id:
                                    topic_url = f"{self.base_url}/t/{topic_id}"
                                    self.logger.info(f"Duplicate post found: {topic_url}")
                                    return True, topic_url
                                
        except Exception as e:
            self.logger.error(f"Error during Discourse API request: {e}")
            return False, None
            
        return False, None
        
    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL by removing query parameters, fragments, and trailing slashes.
        
        Args:
            url (str): The URL to normalize.
            
        Returns:
            str: The normalized URL.
        """
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Reconstruct the URL without query parameters and fragments
        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        
        # Remove trailing slash if present
        if normalized_url.endswith('/'):
            normalized_url = normalized_url[:-1]
            
        return normalized_url
    # Search the discourse api for a query
    async def search_discourse(self, query: str):
        """
        Search Discourse for posts matching a query.
        
        Args:
            query (str): The search query
            
        Returns:
            dict: The search results or None if the search fails
        """
        search_url = f"{self.base_url}/search.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.api_key,
            "Api-Username": self.api_username,
        }
        
        params = {"q": query}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params, timeout=15) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                        return None
                    
                    if response.status == 200:
                        return response_json
                    else:
                        self.logger.error(f"Discourse API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.logger.error(f"Error searching Discourse: {e}")
            return None

    async def get_all_tags(self):
        """
        Get all tags from Discourse.
        
        Returns:
            list: All tags or None if the request fails
        """
        try:
            tags_url = f"{self.base_url}/tags.json"
            headers = {
                "Content-Type": "application/json",
                "Api-Key": self.api_key,
                "Api-Username": self.api_username,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(tags_url, headers=headers) as response:
                    response_text = await response.text()
                    self.logger.debug(f"Response text: {response_text}")  # Log the raw response text
                    
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                        return None
                    
                    self.logger.debug(f"Response JSON: {response_json}")
                    
                    if response.status == 200:
                        if "tags" in response_json:
                            if isinstance(response_json["tags"], list):
                                return response_json["tags"]
                            else:
                                self.logger.error("Unexpected format for 'tags' in response. Expected a list.")
                        else:
                            self.logger.error("Unexpected response structure. 'tags' key not found.")
                        return None
                    else:
                        self.logger.error(f"Discourse API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching tags: {e}")
            return None

async def scrape_twitter_content(url: str) -> Optional[str]:
    """Scrape content from a Twitter URL using snscrape."""
    try:
        import snscrape.modules.twitter as sntwitter
        # Extract tweet ID from URL
        tweet_id = url.split("/status/")[-1].split("?")[0]
        
        # Create scraper instance and get tweet
        scraper = sntwitter.TwitterTweetScraper(tweet_id)
        tweet = None
        
        # Get the tweet data
        for tweet_data in scraper.get_items():
            tweet = tweet_data
            break  # We only need the first (and should be only) tweet
            
        if not tweet:
            logger.error(f"Could not find tweet with ID: {tweet_id}")
            return None
            
        # Format the tweet content
        content = f"""Tweet by @{tweet.user.username}:
{tweet.rawContent}

Posted: {tweet.date}
Likes: {tweet.likeCount}
Retweets: {tweet.retweetCount}
Replies: {tweet.replyCount}"""

        # Add quote tweet content if present
        if tweet.quotedTweet:
            content += f"\n\nQuoted Tweet by @{tweet.quotedTweet.user.username}:\n{tweet.quotedTweet.rawContent}"
            
        # Add media information if present
        if tweet.media:
            media_urls = []
            for media in tweet.media:
                if hasattr(media, 'fullUrl'):
                    media_urls.append(media.fullUrl)
                elif hasattr(media, 'url'):
                    media_urls.append(media.url)
            
            if media_urls:
                content += "\n\nMedia URLs:\n" + "\n".join(media_urls)
        
        return content
        
    except Exception as e:
        logger.error(f"Error scraping Twitter content: {str(e)}")
        return None
async def scrape_linkedin_content(url: str, timeout: int = 10) -> Optional[str]:
    """
    Scrape content from a LinkedIn URL.
    
    Args:
        url (str): The LinkedIn URL to scrape
        timeout (int): Request timeout in seconds
        
    Returns:
        Optional[str]: The scraped content, or None if scraping fails
    """
    try:
        # Use generic scraping for LinkedIn
        return await generic_scrape_content(url)
    except Exception as e:
        logger.error(f"Error scraping LinkedIn content: {str(e)}")
        return None

def extract_linkedin_post_id(url: str) -> Optional[tuple[str, str]]:
    """
    Extract the post ID and content type from various LinkedIn URL formats.
    Supported formats:
      - Posts: linkedin.com/posts/{post_id}
      - Articles: linkedin.com/pulse/{article_id}
      - Company Updates: linkedin.com/company/{company}/posts/{post_id}
      - Profiles: linkedin.com/in/{profile_id}
      
    Args:
        url (str): The LinkedIn URL.
        
    Returns:
        Optional[tuple[str, str]]: (post_id, content_type) if a pattern matches; otherwise, None.
    """
    try:
        patterns = {
            r'linkedin\.com/posts/([a-zA-Z0-9-]+)': 'ugcPosts',
            r'linkedin\.com/pulse/([a-zA-Z0-9-]+)': 'articles',
            r'linkedin\.com/company/[^/]+/posts/([a-zA-Z0-9-]+)': 'companyPosts',
            r'linkedin\.com/in/([a-zA-Z0-9-]+)': 'profiles'
        }

        # Use re.IGNORECASE for case-insensitive matching.
        for pattern, content_type in patterns.items():
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                logger.debug(f"Matched pattern: {pattern} for URL: {url}")
                return (match.group(1), content_type)

        logger.warning(f"URL format not recognized: {url}")
    except Exception as e:
        logger.error(f"Error extracting LinkedIn post ID: {e}")
    return None


async def format_linkedin_content(post_data: dict, content_type: str) -> str:
    """
    Format LinkedIn content based on its type.
    
    Args:
        post_data (dict): JSON data returned by the LinkedIn API.
        content_type (str): Type of content (e.g. 'ugcPosts', 'articles', etc.)
        
    Returns:
        str: A formatted string summarizing the post content.
    """
    try:
        author_name = post_data.get('author', {}).get('name', 'Unknown Author')
        # Try multiple keys for content.
        commentary = post_data.get('commentary') or post_data.get('text') or "No content available"
        created_time = post_data.get('created', {}).get('time', 'Unknown date')
        social_counts = post_data.get('socialDetail', {}).get('totalSocialActivityCounts', {})
        reactions = social_counts.get('reactions', 0)
        comments = social_counts.get('comments', 0)

        base_content = (
            f"Post by {author_name}:\n"
            f"{commentary}\n\n"
            f"Posted: {created_time}\n"
            f"Reactions: {reactions}\n"
            f"Comments: {comments}\n"
        )

        # Handle media attachments if present.
        media_items = post_data.get('content', {}).get('media', [])
        if media_items:
            media_urls = []
            for item in media_items:
                original_url = item.get('originalUrl')
                if original_url:
                    media_urls.append(original_url)
            if media_urls:
                base_content += "\nMedia URLs:\n" + "\n".join(media_urls)
                
        return base_content

    except Exception as e:
        logger.error(f"Error formatting LinkedIn content: {e}")
        return "Error formatting content"
    
async def scrape_reddit_content(url: str) -> Optional[str]:
    """
    Scrape content from a Reddit URL.
    
    Args:
        url (str): The Reddit URL to scrape
        
    Returns:
        Optional[str]: The scraped content, or None if scraping fails
    """
    try:
        # Extract post ID from URL
        post_id = extract_reddit_post_id(url)
        if not post_id:
            return None
        
        # Use Reddit JSON API
        api_url = f"https://www.reddit.com/comments/{post_id}.json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch Reddit content: {response.status}")
                    return None
                
                try:
                    data = await response.json()
                except Exception as e:
                    logger.error(f"Error decoding Reddit response: {e}")
                    return None
                
                if not data or not isinstance(data, list) or len(data) < 1:
                    logger.error("No data found for Reddit post")
                    return None
                
                # Extract post data
                post_data = data[0]["data"]["children"][0]["data"]
                
                # Get title and content
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                author = post_data.get("author", "")
                subreddit = post_data.get("subreddit_name_prefixed", "")
                score = post_data.get("score", 0)
                num_comments = post_data.get("num_comments", 0)
                
                # Format content
                content = f"""Reddit Post: {title}
URL: {url}
Author: u/{author}
Subreddit: {subreddit}
Score: {score}
Comments: {num_comments}

{selftext}
"""
                
                # Add link to media if present
                if post_data.get("url") and "reddit.com" not in post_data["url"]:
                    content += f"\nMedia URL: {post_data['url']}"
                
                # Add top comments if available
                if len(data) > 1 and "data" in data[1] and "children" in data[1]["data"]:
                    comments = data[1]["data"]["children"]
                    if comments:
                        content += "\n\nTop Comments:\n"
                        comment_count = 0
                        for comment in comments:
                            if "data" in comment and "body" in comment["data"]:
                                comment_author = comment["data"].get("author", "")
                                comment_body = comment["data"]["body"]
                                comment_score = comment["data"].get("score", 0)
                                
                                content += f"\nu/{comment_author} ({comment_score} points):\n{comment_body}\n"
                                
                                comment_count += 1
                                if comment_count >= 3:  # Limit to top 3 comments
                                    break
                
                return content
                
    except Exception as e:
        logger.error(f"Error scraping Reddit content: {str(e)}")
        return None

def extract_reddit_post_id(url: str) -> Optional[str]:
    """Extract post ID from various Reddit URL formats."""
    try:
        # Handle various Reddit URL formats
        patterns = [
            r'reddit\.com/r/\w+/comments/(\w+)',  # Standard format
            r'redd\.it/(\w+)',                    # Short format
            r'reddit\.com/comments/(\w+)',        # Direct comment link
            r'reddit\.com/\w+/(\w+)'              # Other variants
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        logger.error(f"No matching pattern found for URL: {url}")
        return None
    except Exception as e:
        logger.error(f"Error extracting Reddit post ID: {str(e)}")
        return None

async def generic_scrape_content(url: str) -> Optional[str]:
    """
    Generic scraper for web content using BeautifulSoup.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        Optional[str]: The scraped content, or None if scraping fails
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 403:
                    logger.warning(f"Access forbidden for URL: {url}")
                    return None
                elif response.status == 404:
                    logger.warning(f"URL not found: {url}")
                    return None
                elif response.status != 200:
                    logger.error(f"Failed to fetch URL: {url}, status: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get title
                title = soup.title.string if soup.title else ""
                
                # Try to get main content
                main_content = ""
                
                # Try common content containers
                content_candidates = soup.select("article, .content, .post, .entry, main, #content, .post-content")
                if content_candidates:
                    # Use the first content container found
                    main_element = content_candidates[0]
                    paragraphs = main_element.find_all("p")
                    main_content = "\n".join(p.get_text().strip() for p in paragraphs)
                
                # If no main content found, get all paragraphs
                if not main_content:
                    paragraphs = soup.find_all("p")[:10]  # Limit to first 10 paragraphs
                    main_content = "\n".join(p.get_text().strip() for p in paragraphs)
                
                # Get meta description
                meta_desc = ""
                meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
                if meta_tag and "content" in meta_tag.attrs:
                    meta_desc = meta_tag["content"]
                
                # Combine all content
                scraped_content = f"""
Title: {title}
URL: {url}
Description: {meta_desc}

Content:
{main_content}
"""
                return scraped_content.strip()
    except aiohttp.ClientError as e:
        logger.error(f"Network error while scraping {url}: {str(e)}")
        return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout while scraping {url}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in generic scraping for {url}: {str(e)}")
        return None

async def scrape_pdf_content(url: str) -> Optional[str]:
    """
    Scrape content from a PDF URL with improved structure detection.
    Extracts text while attempting to preserve headings and paragraphs.
    """
    try:
        # Clean the URL by removing the fragment identifier
        clean_url = url.split('#')[0]

        async with aiohttp.ClientSession() as session:
            async with session.get(clean_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch PDF: {clean_url}, status: {response.status}")
                    return None
                data = await response.read()

        # Use pdfminer.six for text extraction with layout analysis
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from io import BytesIO, StringIO
        
        pdf_file = BytesIO(data)
        output = StringIO()
        
        # Configure LAParams for better structure detection
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=True
        )
        
        # Extract text with layout analysis
        extract_text_to_fp(pdf_file, output, laparams=laparams, page_numbers=[0, 1])  # First two pages
        text = output.getvalue()
        
        # Basic structure detection - identify potential headings
        lines = text.split('\n')
        structured_content = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    structured_content.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
                
            # Potential heading detection (simplified)
            if len(line) < 100 and line.endswith(':') or line.isupper() or line.istitle():
                if current_paragraph:
                    structured_content.append(' '.join(current_paragraph))
                    current_paragraph = []
                structured_content.append(f"## {line}")
            else:
                current_paragraph.append(line)
                
        if current_paragraph:
            structured_content.append(' '.join(current_paragraph))
            
        # Format the extracted content
        formatted_content = f"""
# PDF Content Summary

URL: {url}

""" + '\n\n'.join(structured_content[:5])  # Limit to first 5 sections for brevity
        return formatted_content.strip()
        
    except Exception as e:
        logger.error(f"Error scraping PDF content from {url}: {str(e)}")
        # Fallback to basic extraction if layout analysis fails
        try:
            from pdfminer.high_level import extract_text
            pdf_file = BytesIO(data)
            text = extract_text(pdf_file, page_numbers=[0])
            return f"# PDF Content Summary\n\nURL: {url}\n\n{text[:1000]}..."  # First 1000 chars
        except Exception as inner_e:
            logger.error(f"Fallback PDF extraction failed: {str(inner_e)}")
            return None

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text using a regular expression.
    
    Args:
        text (str): The text to extract URLs from
        
    Returns:
        List[str]: A list of URLs found in the text
    """
    if not text:
        return []
        
    # This regex matches URLs starting with http:// or https://
    # and continuing until whitespace or certain punctuation
    url_regex = r'(https?://[^\s()<>]+(?:\([^\s()<>]+\)|[^\s`!()\[\]{};:\'".,<>?«»""'']))'
    
    # Find all matches
    matches = re.findall(url_regex, text)
    
    # Clean up URLs (remove trailing punctuation that might have been included)
    cleaned_urls = []
    for url in matches:
        # Remove trailing punctuation
        while url and url[-1] in ',.!?:;\'")]}':
            url = url[:-1]
        cleaned_urls.append(url)
    
    return cleaned_urls

# Main plugin class
class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.config.load_and_update()
        # Initialize logger as a class attribute
        self.logger = logger
        logger.info("MatrixToDiscourseBot started")
        
        # Validate configuration before proceeding
        if not self.config.validate_config():
            logger.error("Configuration validation failed. Bot may not function properly.")
        
        self.ai_integration = AIIntegration(self.config, logger)
        self.discourse_api = DiscourseAPI(self.config, logger)
        self.target_audience = self.safe_config_get("target_audience", "community members")
        
        # Dictionary to map Matrix message IDs to Discourse post IDs
        self.message_id_map = {}
        
        # Message processing locks to prevent race conditions
        self._message_locks = {}
        self._lock_cleanup_task = None
        
        # Load existing message ID mappings
        await self.load_message_id_map()
        
        # Start periodic save task with better error handling
        self.periodic_save_task = asyncio.create_task(self.periodic_save())
        
        # Start lock cleanup task
        self._lock_cleanup_task = asyncio.create_task(self._cleanup_old_locks())
        
        # Log configuration status
        logger.info(f"AI model type: {self.safe_config_get('ai_model_type', 'none')}")
        logger.info(f"URL listening enabled: {self.safe_config_get('url_listening', False)}")
        logger.info(f"Discourse base URL: {self.safe_config_get('discourse_base_url', 'not configured')}")
        logger.info("MatrixToDiscourseBot initialization completed")
    
    def safe_config_get(self, key: str, default=None):
        """
        Safely get a value from the config, handling RecursiveDict properly.
        
        Args:
            key (str): The config key to get
            default: The default value to return if the key is not found
            
        Returns:
            The value from the config, or the default value if not found
        """
        try:
            # Check if this is a RecursiveDict (mautrix config type)
            if 'RecursiveDict' in str(type(self.config)):
                # RecursiveDict requires the default value as a positional argument
                try:
                    return self.config.get(key, default)
                except TypeError:
                    # If the signature is different, try with positional args
                    try:
                        if key in self.config:
                            return self.config[key]
                        else:
                            return default
                    except Exception:
                        return default
            elif hasattr(self.config, 'get'):
                # Standard dict-like get method
                return self.config.get(key, default)
            elif isinstance(self.config, dict) and key in self.config:
                return self.config[key]
            else:
                return default
        except Exception as e:
            logger.warning(f"Error getting config value for key {key}: {e}")
            return default

    async def stop(self) -> None:
        logger.info("Stopping MatrixToDiscourseBot...")
        
        # Cancel all background tasks
        tasks_to_cancel = []
        
        # Periodic save task
        if hasattr(self, 'periodic_save_task') and self.periodic_save_task and not self.periodic_save_task.done():
            tasks_to_cancel.append(('periodic_save_task', self.periodic_save_task))
        
        # Lock cleanup task
        if hasattr(self, '_lock_cleanup_task') and self._lock_cleanup_task and not self._lock_cleanup_task.done():
            tasks_to_cancel.append(('_lock_cleanup_task', self._lock_cleanup_task))
        
        # Cancel all tasks
        for task_name, task in tasks_to_cancel:
            logger.debug(f"Cancelling {task_name}")
            task.cancel()
        
        # Wait for all tasks to complete cancellation
        for task_name, task in tasks_to_cancel:
            try:
                await task
            except asyncio.CancelledError:
                logger.debug(f"{task_name} cancelled successfully")
            except Exception as e:
                logger.error(f"Error stopping {task_name}: {e}", exc_info=True)
        
        # Save message ID mappings before stopping
        try:
            await self.save_message_id_map()
            logger.info("Message ID mappings saved successfully")
        except Exception as e:
            logger.error(f"Error saving message ID mappings during shutdown: {e}", exc_info=True)
        
        # Clear message locks
        self._message_locks.clear()
        
        # Close any open HTTP sessions if they exist
        if hasattr(self, 'http') and self.http:
            try:
                await self.http.close()
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")
        
        await super().stop()
        logger.info("MatrixToDiscourseBot stopped successfully")

    async def periodic_save(self) -> None:
        """Periodically save the message ID mapping to ensure we don't lose data."""
        try:
            while True:
                # Save every hour
                await asyncio.sleep(3600)
                try:
                    await self.save_message_id_map()
                    logger.debug("Periodic save completed successfully")
                except Exception as e:
                    logger.error(f"Error during periodic save: {e}")
        except asyncio.CancelledError:
            logger.info("Periodic save task cancelled")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in periodic save task: {e}")
            raise

    async def _cleanup_old_locks(self) -> None:
        """Periodically clean up old message processing locks to prevent memory leaks."""
        try:
            while True:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                current_time = asyncio.get_event_loop().time()
                
                # Remove locks older than 10 minutes
                old_locks = []
                for message_id, (lock, timestamp) in self._message_locks.items():
                    if current_time - timestamp > 600:  # 10 minutes
                        old_locks.append(message_id)
                
                for message_id in old_locks:
                    if message_id in self._message_locks:
                        del self._message_locks[message_id]
                
                if old_locks:
                    logger.debug(f"Cleaned up {len(old_locks)} old message locks")
                    
        except asyncio.CancelledError:
            logger.info("Lock cleanup task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in lock cleanup task: {e}")
            raise

    async def save_message_id_map(self) -> None:
        """Save the message ID mapping to a JSON file."""
        try:
            # Get the data directory path
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Define the file path
            file_path = os.path.join(data_dir, "message_id_map.json")
            
            # Save the mapping to the file
            with open(file_path, "w") as f:
                json.dump(self.message_id_map, f)
                
            logger.info(f"Saved message ID mapping to {file_path}")
        except Exception as e:
            logger.error(f"Error saving message ID mapping: {e}")

    async def load_message_id_map(self) -> None:
        """Load the message ID mapping from a JSON file."""
        try:
            # Get the data directory path
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            
            # Define the file path
            file_path = os.path.join(data_dir, "message_id_map.json")
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Load the mapping from the file
                with open(file_path, "r") as f:
                    self.message_id_map = json.load(f)
                    
                logger.info(f"Loaded message ID mapping from {file_path} with {len(self.message_id_map)} entries")
            else:
                logger.info("No message ID mapping file found, starting with empty mapping")
        except Exception as e:
            logger.error(f"Error loading message ID mapping: {e}")
            self.message_id_map = {}

    # Function to get the configuration class
    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config
    
    # Function to get the database upgrade table
    @classmethod
    def get_db_upgrade_table(cls) -> UpgradeTable:
        return upgrade_table
    # Command to handle the help event
    @command.new(name=lambda self: self.safe_config_get("help_trigger", "help"), require_subcommand=False)
    async def help_command(self, evt: MessageEvent) -> None:
        await self.handle_help(evt)

    async def handle_help(self, evt: MessageEvent) -> None:
        help_trigger = self.safe_config_get("help_trigger", "help")
        post_trigger = self.safe_config_get("post_trigger", "fpost")
        search_trigger = self.safe_config_get("search_trigger", "search")
        url_post_trigger = self.safe_config_get("url_post_trigger", "url")
        
        logger.info(f"Command !{help_trigger} triggered.")
        help_msg = (
            "Welcome to the Community Forum Bot!\n\n"
            f"To create a post on the forum, reply to a message with `!{post_trigger}`.\n"
            f"To summarize the last N messages, use `!{post_trigger} -n <number>`.\n"
            f"To summarize messages from a timeframe, use `!{post_trigger} -h <hours> -m <minutes> -d <days>`.\n"
            f"To search the forum, use `!{search_trigger} <query>`.\n"
            f"To post a URL, reply to a message containing a URL with `!{url_post_trigger}`.\n"
            f"To reply to a forum post, simply reply to the bot's message that created the post.\n\n"
            f"Admin commands:\n"
            f"- `!save-mappings`: Save the current message ID mappings.\n"
            f"- `!list-mappings`: List the current message ID mappings.\n"
            f"- `!clear-mappings`: Clear all message ID mappings.\n\n"
            f"For help, use `!{help_trigger}`."
        )
        await evt.reply(help_msg)
    async def handle_search_discourse(self, evt: MessageEvent, query: str) -> None:
        search_trigger = self.config.get("search_trigger", "search")
        logger.info(f"Command !{search_trigger} triggered.")
        await evt.reply("Searching the forum and wiki...")

        try:
            # Forum search via Discourse API
            search_results = await self.discourse_api.search_discourse(query)
            
            # Perform MediaWiki search
            mediawiki_base_url = self.config.get("mediawiki_base_url", "")
            wiki_results = await self.search_mediawiki(query)
            top_wiki = wiki_results[:2] if wiki_results else []
            
            if search_results is not None:
                if search_results:
                    # Process forum results
                    for result in search_results:
                        result["views"] = result.get("views", 0)
                        result["created_at"] = result.get("created_at", "1970-01-01T00:00:00Z")
                    
                    # Sort forum posts by views
                    sorted_by_views = sorted(search_results, key=lambda x: x["views"], reverse=True)
                    top_seen = sorted_by_views[:3]
                    
                    def format_forum_results(results):
                        return "\n".join(
                            [
                                f"* [{result['title']}]({self.config['discourse_base_url']}/t/{result['slug']}/{result['id']})"
                                for result in results
                            ]
                        )
                    
                    def format_wiki_results(results):
                        return "\n".join(
                            [
                                f"* [{result['title']}]({mediawiki_base_url}/?curid={result['pageid']})"
                                for result in results
                            ]
                        )
                    
                    result_msg = (
                        "**Wiki Results:**\n"
                        + format_wiki_results(top_wiki)
                        + "\n\n**Top 3 Most Seen (Forum):**\n"
                        + format_forum_results(top_seen)
                    )
                    await evt.reply(f"Search results:\n{result_msg}")
                else:
                    await evt.reply("No forum posts found for the query.")
            else:
                await evt.reply("Failed to perform forum search.")
        except Exception as e:
            logger.error(f"Error processing !{self.config['search_trigger']} command: {e}")
            await evt.reply(f"An error occurred: {e}")

    async def _get_message_lock(self, message_id: str) -> asyncio.Lock:
        """Get or create a lock for message processing to prevent race conditions."""
        current_time = asyncio.get_event_loop().time()
        
        if message_id not in self._message_locks:
            self._message_locks[message_id] = (asyncio.Lock(), current_time)
        
        return self._message_locks[message_id][0]

    # Handle messages with URLs and process them if url_listening is enabled.
    @event.on(EventType.ROOM_MESSAGE)
    async def handle_message(self, evt: MessageEvent) -> None:
        # If URL listening is disabled, don't process URL patterns.
        if not self.safe_config_get("url_listening", False):
            return

        if evt.sender == self.client.mxid:
            return

        if evt.content.msgtype != MessageType.TEXT:
            return

        # Get message lock to prevent race conditions
        message_lock = await self._get_message_lock(evt.event_id)
        
        # Check if message is already being processed
        if message_lock.locked():
            logger.debug(f"Message {evt.event_id} is already being processed, skipping")
            return

        async with message_lock:
            try:
                # Extract URLs from the message body.
                message_body = evt.content.body
                urls = extract_urls(message_body)
                if urls:
                    # Process the message directly instead of calling process_link
                    # This prevents extracting URLs twice
                    try:
                        username = evt.sender.split(":")[0] if ":" in evt.sender else evt.sender  # Extract the username from the sender
                    except (AttributeError, IndexError):
                        username = str(evt.sender)
                    
                    # Filter URLs based on patterns and blacklist
                    urls_to_process = [url for url in urls if self.config.should_process_url(url)]

                    if not urls_to_process:
                        logger.debug("No URLs matched the configured patterns or all URLs were blacklisted")
                        return

                    # Process each URL only once
                    for url in urls_to_process:
                        await self.process_single_url(evt, message_body, url)
            except Exception as e:
                logger.error(f"Error processing message {evt.event_id}: {e}", exc_info=True)

    # Command to process URLs in replies
    @command.new(name=lambda self: self.safe_config_get("url_post_trigger", "url"), require_subcommand=False)
    async def post_url_to_discourse_command(self, evt: MessageEvent) -> None:
        await self.handle_post_url_to_discourse(evt)

    async def handle_post_url_to_discourse(self, evt: MessageEvent) -> None:
        logger.info(f"Command !{self.config.get('url_post_trigger', 'url')} triggered.")

        relation = evt.content.get_reply_to()
        if not relation:
            await evt.reply("You must reply to a message containing a URL to use this command.")
            return

        # Log the relation object for debugging
        logger.debug(f"Relation object: {relation}")
        logger.debug(f"Relation inspection: {self.inspect_relation_object(relation)}")

        # Extract the event ID from the relation object
        replied_to_event_id = await self.extract_event_id_from_relation(relation)
        if not replied_to_event_id:
            await evt.reply("Could not determine which message you're replying to.")
            return

        # Get the replied-to message
        try:
            replied_to_event = await self.client.get_event(evt.room_id, replied_to_event_id)
            if not replied_to_event:
                await evt.reply("Could not retrieve the message you're replying to.")
                return

            # Extract the message body
            if hasattr(replied_to_event, 'content') and 'body' in replied_to_event.content:
                message_body = replied_to_event.content['body']
            else:
                await evt.reply("The message you're replying to doesn't have a body.")
                return

            # Extract URLs from the message body
            urls = extract_urls(message_body)
            if not urls:
                await evt.reply("No URLs found in the message you're replying to.")
                return

            # Process each URL individually
            for url in urls:
                if self.config.should_process_url(url):
                    await self.process_single_url(evt, message_body, url)
                else:
                    logger.debug(f"URL {url} does not match any patterns or is blacklisted.")
            
        except Exception as e:
            logger.error(f"Error handling URL post command: {e}", exc_info=True)
            await evt.reply(f"Error processing URL: {str(e)}")

    async def process_link(self, evt: MessageEvent, message_body: str) -> None:
        urls = extract_urls(message_body)
        try:
            username = evt.sender.split(":")[0] if ":" in evt.sender else evt.sender  # Extract the username from the sender
        except (AttributeError, IndexError):
            username = str(evt.sender)
        
        # Filter URLs based on patterns and blacklist
        urls_to_process = [url for url in urls if self.config.should_process_url(url)]

        if not urls_to_process:
            logger.debug("No URLs matched the configured patterns or all URLs were blacklisted")
            return

        for url in urls_to_process:
            # Check for duplicates
            duplicate_exists, duplicate_url = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"This URL has already been posted, a post summarizing and linking to 12ft.io and archive.org is already on the forum here: {duplicate_url}")
                continue

            try:
                # Initialize AI integration
                ai_integration = AIIntegration(self.config, self.log)  # Pass self.log here

                # Scrape content from the URL
                try:
                    content = await scrape_content(url)
                    if content:
                        logger.info(f"Successfully scraped content from {url} (length: {len(content)} chars)")
                    else:
                        logger.warning(f"Failed to scrape content from {url}")
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    # Continue with a default message
                    content = f"Could not scrape content from {url}. Please visit the link directly."

                # Generate title
                title = await self.generate_title_for_bypassed_links(message_body)
                if not title:
                    logger.info(f"Generating title using URL and domain for: {url}")
                    try:
                        domain = urlparse(url).netloc or url.split('/')[2] if len(url.split('/')) > 2 else url
                        title = await self.generate_title_for_bypassed_links(f"URL: {url}, Domain: {domain}")
                        if not title:
                            title = f"Link from {domain}"
                    except (IndexError, ValueError):
                        title = f"Link from {url}"
                
                # Ensure title is within Discourse's 255 character limit
                if title and len(title) > 250:
                    title = title[:247] + "..."
                
                # Generate tags
                tags = await self.ai_integration.generate_tags(content)
                # Log the generated tags for debugging
                logger.debug(f"Generated tags: {tags}")
                
                # Ensure tags are valid
                if tags is None:
                    tags = ["posted-link"]
                else:
                    # Filter out any invalid tags (empty strings, None values, or tags with invalid characters)
                    tags = [tag for tag in tags if tag and isinstance(tag, str) and re.match(r'^[a-z0-9\-]+$', tag)]
                    
                    # Ensure 'posted-link' is in the tags
                    if "posted-link" not in tags:
                        tags.append("posted-link")
                    
                    # If all tags were filtered out, use default
                    if not tags:
                        tags = ["posted-link"]
                    
                    # Limit to 5 tags maximum
                    tags = tags[:5]
                    
                    # Final check to ensure posted-link is still in the tags after limiting
                    if "posted-link" not in tags:
                        tags[-1] = "posted-link"  # Replace the last tag with posted-link if it got filtered out

                # Generate bypass links
                bypass_links = generate_bypass_links(url)

                # Check which archive links are working
                working_archive_links = {}
                for name, archive_url in bypass_links.items():
                    if name == "original":
                        working_archive_links[name] = archive_url
                        continue
                    
                    try:
                        timeout = aiohttp.ClientTimeout(total=5)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.head(archive_url) as response:
                                if response.status < 400:  # Any successful status code
                                    working_archive_links[name] = archive_url
                    except Exception:
                        logger.debug(f"Archive link {name} is not working for {url}")
                        continue

                # Prepare archive links section
                archive_links_section = "**Archive Links:**\n"
                if len(working_archive_links) <= 1:  # Only original link is working
                    archive_links_section += "No working archive links found. The content may be too recent or behind a paywall.\n"
                else:
                    for name, archive_url in working_archive_links.items():
                        if name != "original":
                            archive_links_section += f"**{name}:** {archive_url}\n"

                # Prepare message body
                post_body = (
                    f"{content}\n\n"
                    f"{archive_links_section}\n"
                    f"**Original Link:** <{url}>\n\n"
                    f"User Message: {message_body}\n\n"
                    f"For more on bypassing paywalls, see the [post on bypassing methods](https://forum.irregularchat.com/t/bypass-links-and-methods/98?u=sac)"
                )

                # Determine category ID based on room ID using the safe_config_get helper
                try:
                    # Get the mapping dictionary with a default empty dict
                    matrix_to_discourse_topic = self.config.get("matrix_to_discourse_topic", {})
                    
                    # Get the unsorted category ID with a default value
                    unsorted_category_id = self.config.get("unsorted_category_id", 1)  # Default to category ID 1 if not set
                    
                    # Get the category ID for this room with the unsorted category ID as default
                    if isinstance(matrix_to_discourse_topic, dict):
                        # Regular dict
                        category_id = matrix_to_discourse_topic.get(evt.room_id, unsorted_category_id)
                    else:
                        # RecursiveDict or other object with get method
                        try:
                            category_id = matrix_to_discourse_topic.get(evt.room_id, unsorted_category_id)
                        except TypeError:
                            # If the get method requires more arguments, try with a default value
                            logger.debug("RecursiveDict detected for matrix_to_discourse_topic, using default value")
                            category_id = unsorted_category_id
                        
                    logger.info(f"Using category ID: {category_id}")
                    logger.info(f"Final tags being used: {tags}")
                except Exception as e:
                    logger.error(f"Error determining category ID: {e}", exc_info=True)
                    # Default to a safe value if we can't determine the category ID
                    category_id = 1  # Default to category ID 1
                    logger.info(f"Using default category ID: {category_id}")

                # Create the post on Discourse
                post_url, error = await self.discourse_api.create_post(
                    title=title,
                    raw=post_body,
                    category_id=category_id,
                    tags=tags,
                )

                if post_url:
                    # Extract topic ID from the post URL
                    try:
                        topic_id = post_url.split("/")[-1]
                        if topic_id.isdigit():
                            # Store the mapping between Matrix message ID and Discourse topic ID
                            self.message_id_map[evt.event_id] = topic_id
                            logger.info(f"Stored mapping: Matrix ID {evt.event_id} -> Discourse topic {topic_id}")
                        else:
                            logger.warning(f"Invalid topic ID extracted from URL: {topic_id}")
                    except (IndexError, AttributeError):
                        logger.error(f"Failed to extract topic ID from post URL: {post_url}")
                    
                    posted_link_url = f"{self.discourse_api.base_url}/tag/posted-link"
                    # post_url should not be markdown
                    post_url = post_url.replace("[", "").replace("]", "")
                    
                    # Get summary length with a safe default
                    summary_length = self.config.get("summary_length_in_characters", 200)
                    
                    # Get the URL post trigger for the reply instruction
                    url_post_trigger = self.config.get("url_post_trigger", "furl")
                    
                    await evt.reply(
                        f"🔗 {title}\n\n"
                        f"_Reply to this message to add a comment directly to the forum post.✅ Will Confirm_\n\n"
                        f"**Forum Post URL:** {post_url}\n\n"
                        f"{content[:summary_length]}...\n\n"
                        
                    )
                else:
                    await evt.reply(f"Failed to create post: {error}")
            except Exception as e:
                logger.error(f"Error processing link: {e}", exc_info=True)

    async def search_mediawiki(self, query: str) -> Optional[List[Dict]]:
        """
        Search the MediaWiki API using the configured base URL.
        Tries both /w/api.php and /api.php, returning a list of search result dictionaries or None on failure.
        """
        mediawiki_base_url = self.config.get("mediawiki_base_url", None)
        if not mediawiki_base_url:
            logger.error("MediaWiki base URL is not configured.")
            return None

        # Prepare candidate endpoint URLs.
        candidate_urls = []
        if "api.php" in mediawiki_base_url:
            candidate_urls.append(mediawiki_base_url)
        else:
            if mediawiki_base_url.endswith("/"):
                candidate_urls.append(mediawiki_base_url + "w/api.php")
                candidate_urls.append(mediawiki_base_url + "api.php")
            else:
                candidate_urls.append(mediawiki_base_url + "/w/api.php")
                candidate_urls.append(mediawiki_base_url + "/api.php")
        
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        headers = {
            "User-Agent": "MatrixToDiscourseBot/1.0"
        }
        
        for candidate_url in candidate_urls:
            logger.debug(f"Trying MediaWiki API candidate URL: {candidate_url} with params: {params}")
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    async with session.get(candidate_url, params=params) as response:
                        if response.status == 200:
                            resp_json = await response.json()
                            logger.debug(f"Candidate URL {candidate_url} returned: {resp_json}")
                            # Check if the response has the expected structure.
                            if "query" in resp_json and "search" in resp_json["query"]:
                                return resp_json["query"]["search"]
                            else:
                                logger.debug(f"Candidate URL {candidate_url} did not return expected keys: {resp_json}")
                        else:
                            body = await response.text()
                            logger.debug(f"Candidate URL {candidate_url} failed with status: {response.status} and body: {body}")
            except Exception as e:
                logger.error(f"Error searching MediaWiki at candidate URL {candidate_url}: {e}")
        
        logger.error("All candidate URLs failed for MediaWiki search.")
        return None

    @event.on(EventType.ROOM_MESSAGE)
    async def handle_matrix_reply(self, evt: MessageEvent) -> None:
        """
        Handle Matrix replies and post them to Discourse as replies.
        
        This method checks if a message is a reply to a previously posted message
        that has a corresponding Discourse topic. If so, it posts the reply content
        to that Discourse topic.
        """
        try:
            # Ignore messages from the bot itself
            if evt.sender == self.client.mxid:
                return
            
            # Check if message is a reply
            relation = evt.content.get_reply_to()
            if not relation:
                return  # Not a reply
            
            # Log the relation object for debugging
            logger.debug(f"Matrix reply - Relation object: {relation}")
            logger.debug(f"Matrix reply - Relation inspection: {self.inspect_relation_object(relation)}")
            
            # Get the event ID of the message being replied to
            replied_event_id = await self.extract_event_id_from_relation(relation)
            if not replied_event_id:
                logger.warning(f"Could not extract event ID from relation: {relation}")
                return
            
            # Log the relation and event ID for debugging
            logger.debug(f"Matrix reply - Relation: {relation}, type: {type(relation)}")
            logger.debug(f"Matrix reply - Extracted replied_event_id: {replied_event_id}")
            
            # Check if we have a mapping for this message ID
            discourse_topic_id = self.message_id_map.get(replied_event_id)
            
            # If we don't have a direct mapping, check if this is a reply to a bot message
            # that contains a forum link (which would be a reply to a previous post)
            if not discourse_topic_id:
                logger.debug(f"No direct mapping found for {replied_event_id}, checking if it's a reply to a bot message")
                
                # Try to get the original message to see if it's from the bot and contains a forum link
                try:
                    replied_event = await self.client.get_event(evt.room_id, replied_event_id)
                    if replied_event and replied_event.sender == self.client.mxid:
                        # This is a reply to a bot message, extract the topic ID from the message content
                        content = replied_event.content.body
                        # Look for a Discourse URL pattern in the bot's message
                        discourse_base_url = self.config.get("discourse_base_url", "")
                        if discourse_base_url and discourse_base_url in content:
                            # Extract the topic ID from the URL
                            url_pattern = f"{discourse_base_url}/t/[^/]+/(\\d+)"
                            match = re.search(url_pattern, content)
                            if match:
                                discourse_topic_id = match.group(1)
                                logger.info(f"Found topic ID {discourse_topic_id} in bot message")
                except Exception as e:
                    logger.warning(f"Error checking replied message: {e}")
            
            if not discourse_topic_id:
                logger.debug(f"No Discourse topic found for Matrix message ID: {replied_event_id}")
                return
            
            # Get the content of the reply
            reply_content = evt.content.body
            
            # Remove the quoted part if present (common in Matrix clients)
            if ">" in reply_content:
                # Split by lines and remove those starting with >
                lines = reply_content.split("\n")
                non_quoted_lines = [line for line in lines if not line.strip().startswith(">")]
                reply_content = "\n".join(non_quoted_lines).strip()
            
            # If after removing quotes there's no content, ignore
            if not reply_content:
                logger.debug("Reply contains only quotes, ignoring")
                return
            
            # Try to get the user's display name
            try:
                user_profile = await self.client.get_profile(evt.sender)
                display_name = user_profile.displayname if user_profile and user_profile.displayname else "a community member"
            except Exception as e:
                logger.warning(f"Could not get display name for {evt.sender}: {e}")
                display_name = "a community member"
            
            # Format the reply for Discourse
            formatted_reply = f"**Matrix reply from {display_name}**:\n\n{reply_content}"
            
            # Post the reply to Discourse
            logger.info(f"Posting reply to Discourse topic {discourse_topic_id}")
            post_url, error = await self.discourse_api.create_post(
                title=None,  # No title needed for replies
                raw=formatted_reply,
                category_id=None,  # Not needed for replies
                tags=None,  # Not needed for replies
                topic_id=discourse_topic_id  # Specify the topic to reply to
            )
            
            if error:
                logger.error(f"Failed to post reply to Discourse: {error}")
            else:
                logger.info(f"Successfully posted reply to Discourse: {post_url}")
                await evt.react("✅")  # React to indicate success
        except Exception as e:
            logger.error(f"Error in handle_matrix_reply: {e}", exc_info=True)

    @command.new(name="save-mappings", require_subcommand=False)
    async def save_mappings_command(self, evt: MessageEvent) -> None:
        """Command to manually save the message ID mapping."""
        # Only allow admins to use this command
        if not await self.is_admin(evt.sender):
            await evt.reply("You don't have permission to use this command.")
            return
        
        await self.save_message_id_map()
        await evt.reply(f"Message ID mapping saved with {len(self.message_id_map)} entries.")

    async def is_admin(self, user_id: str) -> bool:
        """Check if a user is an admin."""
        admins = self.config.get("admins", [])
        logger.debug(f"Checking if {user_id} is in admins list: {admins}")
        return user_id in admins

    async def generate_title_for_bypassed_links(self, message_body: str) -> Optional[str]:
        """Generate a title for a message with bypassed links."""
        if not self.ai_integration:
            self.ai_integration = AIIntegration(self.config, logger)
        
        # Use a more direct prompt for title generation
        prompt = f"Create a brief (3-10 word) attention-grabbing title for the following content. Focus only on the main topic. DO NOT use markdown formatting like # or **. Do not include any commentary, source attribution, or explanations. Just provide the title: {message_body}"
        
        title = await self.ai_integration.generate_title(prompt)
        
        # Clean up the title - remove any "Title:" prefix or quotes that the model might add
        if title:
            title = title.replace("Title:", "").strip()
            title = title.strip('"\'')
            
            # Clean any markdown formatting
            title = self.clean_markdown_from_title(title)
            
            # Ensure the title is not too long for Discourse (max 255 chars)
            if len(title) > 250:
                title = title[:247] + "..."
        
        return title

    async def generate_title(self, message_body: str) -> Optional[str]:
        """
        Generate a title for a message.
        
        Args:
            message_body (str): The message body to generate a title from
            
        Returns:
            Optional[str]: The generated title, or None if generation fails
        """
        return await self.generate_title_for_bypassed_links(message_body)

    async def api_call_with_retry(self, func, *args, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"API call failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    @command.new(name="list-mappings", require_subcommand=False)
    async def list_mappings_command(self, evt: MessageEvent) -> None:
        """Command to list the current message ID mappings."""
        # Only allow admins to use this command
        if not await self.is_admin(evt.sender):
            await evt.reply("You don't have permission to use this command.")
            return
        
        if not self.message_id_map:
            await evt.reply("No message ID mappings found.")
            return
        
        # Format the mappings for display
        mappings = []
        for matrix_id, discourse_id in list(self.message_id_map.items())[:10]:  # Limit to 10 entries
            mappings.append(f"Matrix ID: {matrix_id} -> Discourse Topic: {discourse_id}")
        
        total = len(self.message_id_map)
        shown = min(10, total)
        
        message = f"Message ID Mappings ({shown} of {total} shown):\n\n" + "\n".join(mappings)
        if total > 10:
            message += f"\n\n... and {total - 10} more."
        
        await evt.reply(message)

    @command.new(name="clear-mappings", require_subcommand=False)
    async def clear_mappings_command(self, evt: MessageEvent) -> None:
        """Command to clear all message ID mappings."""
        # Only allow admins to use this command
        if not await self.is_admin(evt.sender):
            await evt.reply("You don't have permission to use this command.")
            return
        
        # Ask for confirmation
        await evt.reply("Are you sure you want to clear all message ID mappings? This cannot be undone. Reply with 'yes' to confirm.")
        
        # Wait for confirmation
        try:
            response = await self.client.wait_for_event(
                EventType.ROOM_MESSAGE,
                room_id=evt.room_id,
                sender=evt.sender,
                timeout=60000  # 60 seconds
            )
            
            if response.content.body.lower() == "yes":
                # Clear the mappings
                count = len(self.message_id_map)
                self.message_id_map.clear()
                await self.save_message_id_map()
                await evt.reply(f"Cleared {count} message ID mappings.")
            else:
                await evt.reply("Operation cancelled.")
        except asyncio.TimeoutError:
            await evt.reply("Confirmation timed out. Operation cancelled.")

    async def extract_event_id_from_relation(self, relation) -> Optional[str]:
        """
        Extract the event ID from a relation object.
        
        Args:
            relation: The relation object from evt.content.get_reply_to()
            
        Returns:
            Optional[str]: The event ID if found, None otherwise
        """
        if not relation:
            logger.debug("Relation is None")
            return None
        
        logger.debug(f"Extracting event ID from relation: {relation} (type: {type(relation)})")
        
        # Log detailed inspection of the relation object
        inspection = self.inspect_relation_object(relation)
        logger.debug(f"Relation inspection:\n{inspection}")
        
        # Handle different relation object structures
        if hasattr(relation, 'event_id'):
            # Object-style relation
            event_id = relation.event_id
            logger.debug(f"Found event_id attribute: {event_id}")
            return event_id
        elif isinstance(relation, str):
            # String-style relation (direct event ID)
            logger.debug(f"Relation is a string: {relation}")
            return relation
        else:
            # Try to access as dictionary
            try:
                # Try attribute access first
                if hasattr(relation, 'event_id'):
                    event_id = relation.event_id
                    logger.debug(f"Found event_id attribute: {event_id}")
                    return event_id
                
                # Try dictionary access
                if isinstance(relation, dict) and 'event_id' in relation:
                    event_id = relation['event_id']
                    logger.debug(f"Found event_id key: {event_id}")
                    return event_id
                
                # Try get method - handle RecursiveDict specifically
                if hasattr(relation, 'get'):
                    try:
                        # For RecursiveDict, we need to provide a default value
                        event_id = relation.get('event_id', None)
                        if event_id:
                            logger.debug(f"Found event_id via get method: {event_id}")
                            return event_id
                    except TypeError as e:
                        # If the get method requires more arguments, try with a default value
                        if "missing 1 required positional argument" in str(e):
                            logger.debug("RecursiveDict detected, trying with default value")
                            try:
                                event_id = relation.get('event_id', None)
                                if event_id:
                                    logger.debug(f"Found event_id via get method with default: {event_id}")
                                    return event_id
                            except Exception as inner_e:
                                logger.warning(f"Error using get method with default: {inner_e}")
                        else:
                            raise
                
                # Check for m.in_reply_to structure (Matrix spec)
                if hasattr(relation, 'm.in_reply_to') or (isinstance(relation, dict) and 'm.in_reply_to' in relation):
                    try:
                        # Try to get m.in_reply_to safely
                        if hasattr(relation, 'get'):
                            try:
                                in_reply_to = relation.get('m.in_reply_to', None)
                            except TypeError:
                                # RecursiveDict requires default value
                                in_reply_to = relation.get('m.in_reply_to', None)
                        else:
                            in_reply_to = relation.get('m.in_reply_to')
                        
                        if in_reply_to and isinstance(in_reply_to, dict) and 'event_id' in in_reply_to:
                            event_id = in_reply_to['event_id']
                            logger.debug(f"Found event_id in m.in_reply_to: {event_id}")
                            return event_id
                    except Exception as e:
                        logger.warning(f"Error accessing m.in_reply_to: {e}")
                
                # If we get here, we couldn't find the event ID
                logger.warning(f"Could not extract event ID from relation: {relation} (type: {type(relation)})")
                return None
            except (AttributeError, TypeError) as e:
                logger.warning(f"Error extracting event ID from relation: {e}")
                return None

    def inspect_relation_object(self, relation) -> str:
        """
        Inspect the structure of a relation object for debugging purposes.
        
        Args:
            relation: The relation object to inspect
            
        Returns:
            str: A string representation of the relation object structure
        """
        if relation is None:
            return "None"
            
        result = []
        result.append(f"Type: {type(relation)}")
        
        # Check if it's a string
        if isinstance(relation, str):
            return f"String: {relation}"
        
        # Check for attributes
        if hasattr(relation, '__dict__'):
            try:
                result.append(f"Attributes: {relation.__dict__}")
            except Exception as e:
                result.append(f"Error accessing __dict__: {e}")
        
        # Check for dictionary-like behavior
        if isinstance(relation, dict):
            try:
                result.append(f"Dictionary keys: {list(relation.keys())}")
            except Exception as e:
                result.append(f"Error accessing keys: {e}")
        
        # Check for specific attributes
        for attr in ['event_id', 'rel_type', 'in_reply_to', 'm.in_reply_to']:
            try:
                if hasattr(relation, attr):
                    result.append(f"Has attribute '{attr}': {getattr(relation, attr)}")
                elif isinstance(relation, dict) and attr in relation:
                    result.append(f"Has key '{attr}': {relation[attr]}")
            except Exception as e:
                result.append(f"Error checking attribute '{attr}': {e}")
        
        # Check for methods
        for method in ['get', 'keys', 'values', 'items']:
            try:
                if hasattr(relation, method) and callable(getattr(relation, method)):
                    result.append(f"Has method '{method}'")
                    
                    # Try to call the method if it's 'get'
                    if method == 'get':
                        try:
                            # Try calling get with a key and default value
                            result.append(f"Testing get method: relation.get('event_id', None) = {relation.get('event_id', None)}")
                        except Exception as e:
                            result.append(f"Error calling get method: {e}")
            except Exception as e:
                result.append(f"Error checking method '{method}': {e}")
        
        # Check if it's a RecursiveDict (common in Matrix)
        if 'RecursiveDict' in str(type(relation)):
            result.append("Detected RecursiveDict (requires default value for get method)")
            
            # Try to access some common keys safely
            for key in ['event_id', 'rel_type', 'm.in_reply_to']:
                try:
                    value = relation.get(key, None)
                    result.append(f"RecursiveDict.get('{key}', None) = {value}")
                except Exception as e:
                    result.append(f"Error accessing RecursiveDict key '{key}': {e}")
        
        return "\n".join(result)

    async def process_single_url(self, evt: MessageEvent, message_body: str, url: str) -> None:
        """Process a single URL from a message and post it to Discourse."""
        try:
            # Check for duplicates
            duplicate_exists, duplicate_url = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"This URL has already been posted, a post summarizing and linking to 12ft.io and archive.org is already on the forum here: {duplicate_url}")
                return

            # Initialize AI integration
            ai_integration = AIIntegration(self.config, logger)

            # Scrape content from the URL
            try:
                content = await scrape_content(url)
                if content:
                    logger.info(f"Successfully scraped content from {url} (length: {len(content)} chars)")
                else:
                    logger.warning(f"Failed to scrape content from {url}")
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                # Continue with a default message
                content = f"Could not scrape content from {url}. Please visit the link directly."

            # Generate title
            title = await self.generate_title_for_bypassed_links(message_body)
            if not title:
                logger.info(f"Generating title using URL and domain for: {url}")
                title = await self.generate_title_for_bypassed_links(f"URL: {url}, Domain: {url.split('/')[2]}")
                if not title:
                    title = f"Link from {url.split('/')[2]}"
            
            # Clean up the title - remove any markdown formatting
            title = self.clean_markdown_from_title(title)
            
            # Ensure title is within Discourse's 255 character limit
            if title and len(title) > 250:
                title = title[:247] + "..."
            
            # Generate tags
            tags = await ai_integration.generate_tags(content)
            # Log the generated tags for debugging
            logger.debug(f"Generated tags: {tags}")
            
            # Ensure tags are valid
            if tags is None:
                tags = ["posted-link"]
            else:
                # Filter out any invalid tags (empty strings, None values, or tags with invalid characters)
                tags = [tag for tag in tags if tag and isinstance(tag, str) and re.match(r'^[a-z0-9\-]+$', tag)]
                
                # Ensure 'posted-link' is in the tags
                if "posted-link" not in tags:
                    tags.append("posted-link")
                
                # If all tags were filtered out, use default
                if not tags:
                    tags = ["posted-link"]
                
                # Limit to 5 tags maximum
                tags = tags[:5]
                
                # Final check to ensure posted-link is still in the tags after limiting
                if "posted-link" not in tags:
                    tags[-1] = "posted-link"  # Replace the last tag with posted-link if it got filtered out

            # Generate bypass links
            bypass_links = generate_bypass_links(url)

            # Check which archive links are working
            working_archive_links = {}
            for name, archive_url in bypass_links.items():
                if name == "original":
                    working_archive_links[name] = archive_url
                    continue
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.head(archive_url, timeout=5) as response:
                            if response.status < 400:  # Any successful status code
                                working_archive_links[name] = archive_url
                except Exception:
                    logger.debug(f"Archive link {name} is not working for {url}")
                    continue

            # Generate a concise summary and key points using AI
            summary_prompt = f"Provide a concise executive summary (maximum 300 words) of the following content. Do not include markdown formatting or headers: {content}"
            key_points_prompt = f"Extract exactly 5 key points from the following content. Format each point as a single complete sentence without markdown formatting or numbering. If there aren't 5 clear points, create logical points based on the available content: {content}"
            
            concise_summary = await ai_integration.summarize_content(summary_prompt)
            if not concise_summary or concise_summary == "Not Enough Information to Summarize":
                concise_summary = "No summary available. Please visit the original link for details."
            else:
                concise_summary = self.clean_markdown_from_summary(concise_summary)
                
            key_points = await ai_integration.summarize_content(key_points_prompt)
            if not key_points or key_points == "Not Enough Information to Summarize":
                key_points = "- No key points available\n- Please visit the original link for details"
            else:
                # Clean markdown from key points
                key_points = self.clean_markdown_from_summary(key_points)
                
                # Ensure key points are properly formatted as bullet points
                if not key_points.startswith("- "):
                    # Split by newlines or periods followed by space
                    points = re.split(r'(?:\n+|\.\s+)', key_points)
                    # Filter out empty points and format as bullet points
                    formatted_points = []
                    for point in points:
                        point = point.strip()
                        if point:
                            # Add period if missing
                            if not point.endswith('.'):
                                point += '.'
                            formatted_points.append(f"- {point}")
                    
                    # Join the formatted points
                    key_points = "\n".join(formatted_points)

            # Prepare archive links section
            archive_links_section = "**Archive Links:**\n"
            if len(working_archive_links) <= 1:  # Only original link is working
                archive_links_section += "No working archive links found. The content may be too recent or behind a paywall.\n"
            else:
                for name, archive_url in working_archive_links.items():
                    if name != "original":
                        archive_links_section += f"**{name}:** {archive_url}\n"

            # Prepare message body with the new format
            post_body = (
                f"**Concise Summary:**\n{concise_summary}\n\n"
                f"**Key Points:**\n{key_points}\n\n"
                f"{archive_links_section}\n"
                f"**Original Link:** <{url}>\n\n"
                f"User Message: {message_body}\n\n"
                f"For more on bypassing paywalls, see the [post on bypassing methods](https://forum.irregularchat.com/t/bypass-links-and-methods/98?u=sac)"
            )

            # Determine category ID based on room ID
            try:
                # Get the mapping dictionary with a default empty dict
                matrix_to_discourse_topic = self.config.get("matrix_to_discourse_topic", {})
                
                # Get the unsorted category ID with a default value
                unsorted_category_id = self.config.get("unsorted_category_id", 1)  # Default to category ID 1 if not set
                
                # Get the category ID for this room with the unsorted category ID as default
                if isinstance(matrix_to_discourse_topic, dict):
                    # Regular dict
                    category_id = matrix_to_discourse_topic.get(evt.room_id, unsorted_category_id)
                else:
                    # RecursiveDict or other object with get method
                    try:
                        category_id = matrix_to_discourse_topic.get(evt.room_id, unsorted_category_id)
                    except TypeError:
                        # If the get method requires more arguments, try with a default value
                        logger.debug("RecursiveDict detected for matrix_to_discourse_topic, using default value")
                        category_id = unsorted_category_id
                    
                logger.info(f"Using category ID: {category_id}")
                logger.info(f"Final tags being used: {tags}")
            except Exception as e:
                logger.error(f"Error determining category ID: {e}", exc_info=True)
                # Default to a safe value if we can't determine the category ID
                category_id = 1  # Default to category ID 1
                logger.info(f"Using default category ID: {category_id}")

            # Create the post on Discourse
            post_url, error = await self.discourse_api.create_post(
                title=title,
                raw=post_body,
                category_id=category_id,
                tags=tags,
            )

            if post_url:
                # Extract topic ID from the post URL
                topic_id = post_url.split("/")[-1]
                # Store the mapping between Matrix message ID and Discourse topic ID
                self.message_id_map[evt.event_id] = topic_id
                logger.info(f"Stored mapping: Matrix ID {evt.event_id} -> Discourse topic {topic_id}")
                
                # post_url should not be markdown
                post_url = post_url.replace("[", "").replace("]", "")
                
                # Get summary length with a safe default
                summary_length = self.config.get("summary_length_in_characters", 200)
                
                # Create a clean, simplified message for the Matrix chat
                await evt.reply(
                    f"🔗 {title}\n\n"
                    f"_Reply to this message to add a comment directly to the forum post.✅ Will Confirm_\n\n"
                    f"**Forum Post:** {post_url}\n\n"
                    f"{concise_summary}\n\n"
                )
            else:
                await evt.reply(f"Failed to create post: {error}")
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}", exc_info=True)

    def clean_markdown_from_title(self, title: str) -> str:
        """Remove markdown formatting from a title."""
        if not title:
            return title
            
        # Remove markdown headers (# Header)
        title = re.sub(r'^#+\s+', '', title)
        
        # Remove markdown bold/italic (**bold**, *italic*)
        title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
        title = re.sub(r'\*(.*?)\*', r'\1', title)
        
        # Remove markdown links ([text](url))
        title = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', title)
        
        # Remove backticks (`code`)
        title = re.sub(r'`(.*?)`', r'\1', title)
        
        return title.strip()
        
    def clean_markdown_from_summary(self, summary: str) -> str:
        """Remove markdown formatting from a summary."""
        if not summary:
            return summary
            
        # Remove markdown headers (# Header)
        summary = re.sub(r'^#+\s+', '', summary)
        summary = re.sub(r'\n#+\s+', '\n', summary)
        
        # Remove markdown bold/italic (**bold**, *italic*)
        summary = re.sub(r'\*\*(.*?)\*\*', r'\1', summary)
        summary = re.sub(r'\*(.*?)\*', r'\1', summary)
        
        # Remove markdown links ([text](url))
        summary = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', summary)
        
        # Remove backticks (`code`)
        summary = re.sub(r'`(.*?)`', r'\1', summary)
        
        # Remove HTML tags
        summary = re.sub(r'<[^>]+>', '', summary)
        
        return summary.strip()

async def scrape_content(url: str) -> Optional[str]:
    """
    Scrape content from a URL based on its type.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        Optional[str]: The scraped content, or None if scraping fails
    """
    try:
        # Check URL type and use appropriate scraper
        if url.lower().endswith('.pdf'):
            return await scrape_pdf_content(url)
        elif "x.com" in url or "twitter.com" in url:
            return await scrape_twitter_content(url)
        elif "reddit.com" in url:
            return await scrape_reddit_content(url)
        elif "linkedin.com" in url:
            return await scrape_linkedin_content(url)
        else:
            # Fallback to generic scraping
            content = await generic_scrape_content(url)
            if content:
                return content
            else:
                # If original URL fails, try archive services
                logger.info(f"Original URL {url} failed, trying archive services")
                return await scrape_from_archives(url)
    except Exception as e:
        logger.error(f"Error scraping content from {url}: {str(e)}")
        return None

async def scrape_from_archives(url: str) -> Optional[str]:
    """
    Try to scrape content from various archive services when the original URL fails.
    
    Args:
        url (str): The original URL to find in archives
        
    Returns:
        Optional[str]: The scraped content from archives, or None if scraping fails
    """
    archive_urls = [
        f"https://web.archive.org/web/{url}",
        f"https://archive.is/{url}",
        f"https://archive.ph/{url}",
        f"https://archive.today/{url}",
        f"https://12ft.io/{url}"
    ]
    
    # Try Google Cache as well
    google_cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
    archive_urls.append(google_cache_url)
    
    for archive_url in archive_urls:
        logger.info(f"Trying to scrape from archive: {archive_url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(archive_url, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Get title
                        title = soup.title.string if soup.title else ""
                        
                        # Get main content (first few paragraphs)
                        paragraphs = soup.find_all("p")[:5]  # Get first 5 paragraphs
                        content = "\n".join(p.get_text().strip() for p in paragraphs)
                        
                        if content:
                            # Combine all content
                            scraped_content = f"""
Title: {title} (Retrieved from {archive_url})

Content:
{content}
"""
                            return scraped_content.strip()
        except Exception as e:
            logger.error(f"Error scraping from archive {archive_url}: {str(e)}")
            continue
    
    return None

def generate_bypass_links(url: str) -> Dict[str, str]:
    """
    Generate bypass links for a URL.
    
    Args:
        url (str): The URL to generate bypass links for
        
    Returns:
        Dict[str, str]: A dictionary of bypass links
    """
    links = {
        "original": url,
        "12ft": f"https://12ft.io/{url}",
        "archive.org": f"https://web.archive.org/web/{url}",
        "archive.is": f"https://archive.is/{url}",
        "archive.ph": f"https://archive.ph/{url}",
        "archive.today": f"https://archive.today/{url}",
    }
    return links


