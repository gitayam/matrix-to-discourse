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
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# BeautifulSoup4 since maubot image uses alpine linux
from bs4 import BeautifulSoup
# Configure logging
logger = logging.getLogger(__name__)

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
        helper.copy("discourse_api_key")  # API key for discourse
        helper.copy("discourse_api_username")  # Username for discourse
        helper.copy("discourse_base_url")  # Base URL for discourse
        helper.copy("unsorted_category_id")  # Default category ID
        helper.copy("matrix_to_discourse_topic")  # Room-to-topic ID map

        # Command triggers
        helper.copy("search_trigger") # trigger for the search command
        helper.copy("post_trigger") # trigger for the post command
        helper.copy("help_trigger") # trigger for the help command
        helper.copy("url_listening") # trigger for the url listening command
        helper.copy("url_post_trigger") # trigger for the url post command
        helper.copy("target_audience") # target audience context for ai generation
        helper.copy("summary_length_in_characters") # length of the summary preview
        # Handle URL patterns and blacklist separately
        if "url_patterns" in helper.base:
            self["url_patterns"] = list(helper.base["url_patterns"])
        else:
            self["url_patterns"] = ["https?://.*"]  # Default to match all URLs

        if "url_blacklist" in helper.base:
            self["url_blacklist"] = list(helper.base["url_blacklist"])
        else:
            self["url_blacklist"] = [] 

        # --- New configuration options for MediaWiki ---
        helper.copy("mediawiki_base_url")  # Base URL for the community MediaWiki wiki
        helper.copy("mediawiki_search_trigger")  # Command trigger for MediaWiki wiki search

    def should_process_url(self, url: str) -> bool:
        """
        Check if a URL should be processed based on patterns and blacklist.

        Args:
        url (str): The URL to check
        
        Returns:
            bool: True if the URL should be processed, False otherwise
        """
        # First check blacklist
        for blacklist_pattern in self["url_blacklist"]:
            if re.search(blacklist_pattern, url, re.IGNORECASE):
                logger.debug(f"URL {url} matches blacklist pattern {blacklist_pattern}")
                return False
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
            parsed_url = urlparse(self['discourse_base_url'])
            base_domain = parsed_url.netloc #netloc is the network location of the URL

            # Add base domain and all subdomains to the blacklist
            if base_domain in url:
                logger.debug(f"URL {url} matches base domain {base_domain} of discourse_base_url")
                return False

        # Then check whitelist patterns
        for pattern in self["url_patterns"]:
            if re.search(pattern, url, re.IGNORECASE):
                logger.debug(f"URL {url} matches whitelist pattern {pattern}")
                return True
    
        # If no patterns match, don't process the URL
        return False

# AIIntegration class
class AIIntegration:
    """
    A class to handle AI integration for the plugin.
    Includes methods for generating titles, summaries, and tags.
    """
    def __init__(self, config, log):
        self.config = config
        self.log = log
        self.target_audience = config["target_audience"]
        # Initialize Discourse API self 
        self.discourse_api = DiscourseAPI(self.config, self.log)

    async def generate_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        ai_model_type = self.config["ai_model_type"]

        if ai_model_type == "openai":
            return await self.generate_openai_title(message_body, use_links_prompt)
        elif ai_model_type == "local":
            return await self.generate_local_title(message_body, use_links_prompt)
        elif ai_model_type == "google":
            return await self.generate_google_title(message_body, use_links_prompt)
        else:
            self.log.error(f"Unknown AI model type: {ai_model_type}")
            return None
    #generate_links_title
    async def generate_links_title(self, message_body: str) -> Optional[str]:
        """Create a title focusing on linked content."""
        prompt = f"Create a concise and relevant title for the following content, focusing on the key message and linked content:\n\n{message_body}"
        return await self.generate_title(prompt, use_links_prompt=True)

    async def summarize_content(self, content: str, user_message: str = "") -> Optional[str]:
        prompt = f"""Summarize the following content, including any user-provided context or quotes. If there isn't enough information, please just say 'Not Enough Information to Summarize':\n\nContent:\n{content}\n\nUser Message:\n{user_message}"""
        ai_model_type = self.config["ai_model_type"]
        if ai_model_type == "openai":
            return await self.summarize_with_openai(prompt)
        elif ai_model_type == "local":
            return await self.summarize_with_local_llm(prompt)
        elif ai_model_type == "google":
            return await self.summarize_with_google(prompt)
        else:
            self.log.error(f"Unknown AI model type: {ai_model_type}")
            return None
    #generate_openai_tags
    TAG_PROMPT = """Analyze the following content and suggest 2-5 relevant tags {user_message}.  
        NOW Choose from or be inspired by these existing tags: {tag_list}
        If none of the existing tags fit well, suggest new related tags.
        The tags should be lowercase, use hyphens instead of spaces, and be concise.       

        Return only the tags as a comma-separated list, no explanation needed."""
    # user_message is the message body of the post used for context for tag generation
    async def generate_tags(self, user_message: str = "") -> Optional[List[str]]:
        try:
            if not self.discourse_api:
                self.log.error("Discourse API is not initialized.")
                return ["posted-link"]  # Return default tag instead of None

            # Get existing tags from Discourse for context
            all_tags = await self.discourse_api.get_all_tags()
            if all_tags is None:
                self.log.error("Failed to fetch tags from Discourse API.")
                return ["posted-link"]  # Return default tag

            # Log the type and content of all_tags for debugging
            self.log.debug(f"Type of all_tags: {type(all_tags)}")
            self.log.debug(f"Content of all_tags: {all_tags}")

            # Check if all_tags is a list and contains dictionaries
            if not isinstance(all_tags, list) or not all(isinstance(tag, dict) for tag in all_tags):
                self.log.error("Unexpected format for all_tags. Expected a list of dictionaries.")
                return ["posted-link"]  # Return default tag

            tag_names = [tag["name"] for tag in all_tags]
            tag_list = ", ".join(list(dict.fromkeys(tag_names)))

            # Create prompt with existing tags context accepts tag list and content , pass user message for context
            prompt = self.TAG_PROMPT.format(tag_list=tag_list, user_message=user_message)

            if self.config["ai_model_type"] == "openai":
                try:
                    api_key = self.config.get('openai.api_key', None)
                    if not api_key:
                        self.log.error("OpenAI API key is not configured.")
                        return ["posted-link"]  # Return default tag

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
                                self.log.error(f"Error decoding OpenAI response: {e}\nResponse text: {response_text}")
                                return ["posted-link"]  # Return default tag

                            if response.status == 200:
                                tags_text = response_json["choices"][0]["message"]["content"].strip()
                                tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
                                # Filter out invalid tags
                                tags = [tag for tag in tags if tag and re.match(r'^[a-z0-9\-]+$', tag)]
                                tags = tags[:5]  # Limit to 5 tags
                                
                                # Ensure we have at least one tag
                                if not tags:
                                    return ["posted-link"]
                                return tags
                            else:
                                self.log.error(f"OpenAI API error: {response.status} {response_json}")
                                return ["posted-link"]  # Return default tag
                except Exception as e:
                    tb = traceback.format_exc()
                    self.log.error(f"Error generating tags with OpenAI: {e}\n{tb}")
                    return ["posted-link"]  # Return default tag

            elif self.config["ai_model_type"] == "local":
                headers = {
                    "Content-Type": "application/json"
                }
                
                data = {
                    "prompt": prompt,
                    "max_tokens": self.config.get("local_llm.max_tokens", 100),
                    "temperature": self.config.get("local_llm.temperature", 0.7),
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(self.config["local_llm.api_endpoint"], headers=headers, json=data) as response:
                        response_text = await response.text()
                        try:
                            response_json = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            self.log.error(f"Error decoding Local LLM response: {e}\nResponse text: {response_text}")
                            return ["posted-link"]  # Return default tag

                        if response.status == 200:
                            tags_text = response_json.get("text", "").strip()
                            tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
                            # Filter out invalid tags
                            tags = [tag for tag in tags if tag and re.match(r'^[a-z0-9\-]+$', tag)]
                            tags = tags[:5]  # Limit to 5 tags
                            
                            # Ensure we have at least one tag
                            if not tags:
                                return ["posted-link"]
                            return tags
                        else:
                            self.log.error(f"Local LLM API error: {response.status} {response_json}")
                            return ["posted-link"]  # Return default tag

            elif self.config["ai_model_type"] == "google":
                api_key = self.config.get('google.api_key', None)
                if not api_key:
                    self.log.error("Google API key is not configured.")
                    return ["posted-link"]  # Return default tag

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
                            self.log.error(f"Error decoding Google API response: {e}\nResponse text: {response_text}")
                            return ["posted-link"]  # Return default tag

                        if response.status == 200:
                            tags_text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                            tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
                            # Filter out invalid tags
                            tags = [tag for tag in tags if tag and re.match(r'^[a-z0-9\-]+$', tag)]
                            tags = tags[:5]  # Limit to 5 tags
                            
                            # Ensure we have at least one tag
                            if not tags:
                                return ["posted-link"]
                            return tags
                        else:
                            self.log.error(f"Google API error: {response.status} {response_json}")
                            return ["posted-link"]  # Return default tag

            else:
                self.log.error(f"Unknown AI model type: {self.config['ai_model_type']}")
                return ["posted-link"]  # Return default tag

        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error generating tags: {e}\n{tb}")
            return ["posted-link"]  # Return default tag instead of "fix-me"

    generate_title_prompt = "Create a brief (3-10 word) attention-grabbing title for the {target_audience} for the following post on the community forum: {message_body}"
    generate_links_title_prompt = "Create a brief (3-10 word) attention-grabbing title for the following post on the community forum include the source and title of the linked content: {message_body}"
    # function to generate a title will pass either generate_title_prompt or generate_links_title_prompt based on the use_links_prompt flag
    # Implement the methods for each AI model
    # when called generate_title_prompt or generate_links_title_prompt will be used
    async def generate_openai_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        # Choose the appropriate prompt based on the use_links_prompt flag
        if use_links_prompt:
            prompt = self.generate_links_title_prompt.format(message_body=message_body)
        else:
            prompt = self.generate_title_prompt.format(target_audience=self.target_audience, message_body=message_body)
        try:
            api_key = self.config.get('openai.api_key', None)
            if not api_key:
                self.log.error("OpenAI API key is not configured.")
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
                        self.log.error(f"Error decoding OpenAI response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        self.log.error(f"OpenAI API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error generating title with OpenAI: {e}\n{tb}")
            return None

    async def summarize_with_openai(self, content: str) -> Optional[str]:
        prompt = f"""Please provide a concise summary of the following content identifying key points and references and a brief executive summary.
        if there isn't enough information please just say 'Not Enough Information to Summarize':\n\n{content}"""
        try:
            api_key = self.config.get('openai.api_key', None)
            if not api_key:
                self.log.error("OpenAI API key is not configured.")
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
                        self.log.error(f"Error decoding OpenAI response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        self.log.error(f"OpenAI API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error summarizing with OpenAI: {e}\n{tb}")
            return None

    async def generate_local_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        #  local LLM API
        if use_links_prompt:
            prompt = self.generate_links_title_prompt.format(message_body=message_body)
        else:
            prompt = self.generate_title_prompt.format(target_audience=self.target_audience, message_body=message_body)
        try:
            pass # Implement local LLM API
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error generating title with local LLM: {e}\n{tb}")
            return None

    async def summarize_with_local_llm(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary which is relevant to the {self.target_audience} of the following content:\n\n{content}"
        try:
            pass # Implement local LLM API
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error summarizing with local LLM: {e}\n{tb}")
            return None

    async def generate_google_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
        if use_links_prompt:
            prompt = self.generate_links_title_prompt.format(message_body=message_body)
        else:
            prompt = self.generate_title_prompt.format(target_audience=self.target_audience, message_body=message_body)
        try:
            api_key = self.config.get('google.api_key', None)
            if not api_key:
                self.log.error("Google API key is not configured.")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "model": self.config["google.model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config["google.max_tokens"],
                "temperature": self.config["google.temperature"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["google.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.log.error(f"Error decoding Google API response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        # Adjust according to actual response format
                        return response_json["candidates"][0]["output"].strip()
                    else:
                        self.log.error(f"Google API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error generating title with Google: {e}\n{tb}")
            return None

    async def summarize_with_google(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary which is relevant to the {self.target_audience} of the following content:\n\n{content}"
        try:
            api_key = self.config.get('google.api_key', None)
            if not api_key:
                self.log.error("Google API key is not configured.")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "model": self.config["google.model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config["google.max_tokens"],
                "temperature": self.config["google.temperature"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["google.api_endpoint"], headers=headers, json=data) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.log.error(f"Error decoding Google API response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        # Adjust according to actual response format
                        return response_json["candidates"][0]["output"].strip()
                    else:
                        self.log.error(f"Google API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error summarizing with Google: {e}\n{tb}")
            return None

# DiscourseAPI class
class DiscourseAPI:
    def __init__(self, config, log):
        self.config = config
        self.log = log
    #Get top tags used with discourse api
    async def create_post(self, title, raw, category_id, tags=None):
        url = f"{self.config['discourse_base_url']}/posts.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        # Log the payload
        payload = {
            "title": title,
            "raw": raw,
            "category": category_id,
            "tags": tags or [],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_text = await response.text()
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                    return None, f"Error decoding Discourse response: {e}"

                if response.status == 200:
                    topic_id = data.get("topic_id")
                    topic_slug = data.get("topic_slug")
                    post_url = (
                        f"{self.config['discourse_base_url']}/t/{topic_slug}/{topic_id}"
                        if topic_id and topic_slug
                        else "URL not available"
                    )
                    return post_url, None
                else:
                    self.log.error(f"Discourse API error: {response.status} {data}")
                    return None, f"Failed to create post: {response.status}\nResponse: {response_text}"
    # Check for duplicate posts with discourse api
    async def check_for_duplicate(self, url: str, tag: str = "posted-link") -> bool:
        """
        Check for duplicate posts by searching for a URL within posts.

        Args:
            url (str): The URL to check for duplicates.
            tag (str): The tag to filter posts. Defaults to "posted-link".

        Returns:
            bool: True if a duplicate post is found, False otherwise.
            str: The URL of the duplicate post if found, None otherwise.
        """
        search_url = f"{self.config['discourse_base_url']}/search/query.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        
        # Normalize the URL for more accurate matching
        normalized_url = self.normalize_url(url)
        self.log.debug(f"Checking for duplicates of normalized URL: {normalized_url}")

        # First try searching directly for the URL
        params = {"term": normalized_url}
        self.log.debug(f"Searching Discourse with direct URL: {params}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        self.log.error(f"Discourse API error: {response.status}")
                        return False, None

                    response_json = await response.json()
                    self.log.debug(f"Received response for direct URL search: {response_json}")

                    # Check if any posts contain the URL
                    if response_json.get("posts", []):
                        for post in response_json.get("posts", []):
                            post_id = post.get("id")
                            topic_id = post.get("topic_id")
                            if topic_id:
                                topic_url = f"{self.config['discourse_base_url']}/t/{topic_id}"
                                self.log.info(f"Duplicate post found: {topic_url}")
                                return True, topic_url
            
            # If no direct match, try searching for posts with the tag
            params = {"term": f'tags:{tag}'}
            self.log.debug(f"Searching Discourse with tag: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        self.log.error(f"Discourse API error: {response.status}")
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
                                    topic_url = f"{self.config['discourse_base_url']}/t/{topic_id}"
                                    self.log.info(f"Duplicate post found: {topic_url}")
                                    return True, topic_url
                                
        except Exception as e:
            self.log.error(f"Error during Discourse API request: {e}")
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
        search_url = f"{self.config['discourse_base_url']}/search.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        params = {"q": query}

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, params=params) as response:
                response_text = await response.text()
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError as e:
                    self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                    return None

                if response.status == 200:
                    return response_json.get("topics", [])
                else:
                    self.log.error(f"Discourse API error: {response.status} {response_json}")
                    return None
    # get_top_tags from discourse api to use for tags
    async def get_top_tags(self):
        """Fetch top tags from Discourse API.
        
        Returns:
            dict: JSON response containing tags information, or None if the request fails
        """
        #var with number of tags to get
        num_tags = 10
        url = f"{self.config['discourse_base_url']}/tags.json" # get tags from discourse api
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        # using try because the api key may be invalid
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        # return top num_tags tags
                        return response_json[:num_tags]
                    else:
                        self.log.error(f"Discourse API error: {response.status} {response_json}")
                        return None
        # Log the error if there is one
        except Exception as e:
            self.log.error(f"Error fetching top tags: {e}")
            return None

    async def get_all_tags(self):
        """Fetch all tags from Discourse API.
        
        Returns:
            list: A list of all tag names, or None if the request fails
        """
        url = f"{self.config['discourse_base_url']}/tags.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    response_text = await response.text()
                    self.log.debug(f"Response text: {response_text}")  # Log the raw response text
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        # Log the entire JSON response for debugging
                        self.log.debug(f"Response JSON: {response_json}")
                        # Check if the response contains a 'tags' key with a list of tags
                        if isinstance(response_json, dict) and "tags" in response_json:
                            tags = response_json["tags"]
                            if isinstance(tags, list):
                                # Extract all tag names
                                all_tags = [tag for tag in tags if isinstance(tag, dict) and "name" in tag]
                                return all_tags
                            else:
                                self.log.error("Unexpected format for 'tags' in response. Expected a list.")
                                return None
                        else:
                            self.log.error("Unexpected response structure. 'tags' key not found.")
                            return None
                    else:
                        self.log.error(f"Discourse API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.log.error(f"Error fetching tags: {e}")
            return None

# URL handling functions
def extract_urls(text: str) -> List[str]:
    url_regex = r'(https?://\S+)'
    return re.findall(url_regex, text)
# Generate bypass links for the url
def generate_bypass_links(url: str) -> Dict[str, str]:
    links = {
        "original": url,
        "12ft": f"https://12ft.io/{url}",
        "archive.org": f"https://web.archive.org/web/{url}",
        "archive.is": f"https://archive.is/{url}",
        "archive.ph": f"https://archive.ph/{url}",
        "archive.today": f"https://archive.today/{url}",
    }
    return links

async def scrape_content(url: str) -> Optional[str]:
    """Scrape content from URL using BeautifulSoup or specific APIs for social media."""
    try:
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
    """Try to scrape content from various archive services when the original URL fails."""
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
    Scrape content from a LinkedIn URL using direct HTML scraping instead of the API.
    This method doesn't require authentication and works with public LinkedIn posts.
    
    Args:
        url (str): The LinkedIn URL to scrape.
        timeout (int): Request timeout in seconds.
        
    Returns:
        Optional[str]: The formatted content or None on failure.
    """
    try:
        # Use custom headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
            "Referer": "https://www.google.com/"
        }
        
        # Use a timeout for the request
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch LinkedIn URL: {url}, status: {response.status}")
                    # Try archive services as a fallback
                    return await scrape_from_archives(url)
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract post information
                title = soup.title.string if soup.title else "LinkedIn Post"
                
                # Try to extract author information
                author = None
                author_elem = soup.select_one('a.share-update-card__actor-text-link')
                if author_elem:
                    author = author_elem.get_text(strip=True)
                
                # Try to extract post content
                content = ""
                content_elem = soup.select_one('div.share-update-card__update-text')
                if content_elem:
                    content = content_elem.get_text(strip=True)
                
                # If we couldn't find content through specific selectors, try more generic approaches
                if not content:
                    # Look for article content
                    article_elem = soup.select_one('article')
                    if article_elem:
                        paragraphs = article_elem.select('p')
                        content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                
                # If still no content, extract meta description
                if not content:
                    meta_desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
                    if meta_desc:
                        content = meta_desc.get("content", "")
                
                # Format the extracted information
                formatted_content = f"LinkedIn Post\n\n"
                
                if author:
                    formatted_content += f"Author: {author}\n\n"
                
                formatted_content += f"Content:\n{content}\n\n"
                formatted_content += f"URL: {url}"
                
                return formatted_content
                
    except Exception as e:
        logger.error(f"Error scraping LinkedIn content: {str(e)}")
        # Try archive services as a fallback
        return await scrape_from_archives(url)

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
    """Scrape content from a Reddit URL including post details, media, and metadata."""
    try:
        # Extract post ID from various Reddit URL formats
        post_id = extract_reddit_post_id(url)
        if not post_id:
            logger.error(f"Could not extract post ID from URL: {url}")
            return None

        headers = {"User-Agent": "MatrixToDiscourseBot/1.0"}
        api_url = f"https://api.reddit.com/api/info/?id=t3_{post_id}"
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(api_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch Reddit content: {response.status}")
                    return None

                try:
                    data = await response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding Reddit response: {e}")
                    return None

                if not data.get('data', {}).get('children'):
                    logger.error("No data found for Reddit post")
                    return None
                
                post_data = data['data']['children'][0]['data']
                
                # Build formatted content with comprehensive post information
                content_parts = []
                
                # Title and basic info
                content_parts.append(f"Title: {post_data.get('title', 'No Title')}")
                content_parts.append(f"Author: u/{post_data.get('author', '[deleted]')}")
                content_parts.append(f"Subreddit: r/{post_data.get('subreddit', 'unknown')}")
                
                # Post statistics
                stats = (
                    f"Score: {post_data.get('score', 0)} | "
                    f"Upvote Ratio: {post_data.get('upvote_ratio', 0) * 100:.0f}% | "
                    f"Comments: {post_data.get('num_comments', 0)}"
                )
                content_parts.append(stats)
                
                # Main content
                if post_data.get('selftext'):
                    content_parts.append("\nContent:")
                    content_parts.append(post_data['selftext'])
                
                # Handle media content
                if post_data.get('is_video') and post_data.get('media', {}).get('reddit_video'):
                    content_parts.append("\nVideo URL:")
                    content_parts.append(post_data['media']['reddit_video']['fallback_url'])
                
                if post_data.get('url') and post_data.get('post_hint') == 'image':
                    content_parts.append("\nImage URL:")
                    content_parts.append(post_data['url'])
                
                # Handle crosspost
                if post_data.get('crosspost_parent_list'):
                    crosspost = post_data['crosspost_parent_list'][0]
                    content_parts.append("\nCrossposted from:")
                    content_parts.append(f"r/{crosspost.get('subreddit')} - {crosspost.get('title')}")
                
                # Handle external links
                if post_data.get('url') and not post_data.get('is_self'):
                    content_parts.append("\nExternal Link:")
                    content_parts.append(post_data['url'])
                
                return "\n".join(content_parts)
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
    """Generic scraping for non-social media URLs with improved headers and error handling."""
    # Common user agents to rotate through
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
    ]
    
    # Custom headers to mimic a browser
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "TE": "Trailers",
        "Referer": "https://www.google.com/"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=15, allow_redirects=True) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch URL: {url}, status: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Get title
                title = soup.title.string if soup.title else ""
                
                # Get meta description
                description = ""
                meta_desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
                if meta_desc:
                    description = meta_desc.get("content", "")
                
                # Get meta keywords
                keywords = ""
                meta_keywords = soup.find("meta", {"name": "keywords"})
                if meta_keywords:
                    keywords = meta_keywords.get("content", "")
                
                # Try different content extraction strategies
                content = ""
                
                # Strategy 1: Look for article or main content
                article = soup.find("article") or soup.find("main") or soup.find(id=["content", "main", "article"])
                if article:
                    paragraphs = article.find_all("p")
                    content = "\n".join(p.get_text().strip() for p in paragraphs[:10])  # Get up to 10 paragraphs
                
                # Strategy 2: If no article found, try common content containers
                if not content:
                    for container in ["div.content", "div.article", "div.post", ".post-content", ".article-content"]:
                        content_div = soup.select_one(container)
                        if content_div:
                            paragraphs = content_div.find_all("p")
                            content = "\n".join(p.get_text().strip() for p in paragraphs[:10])
                            break
                
                # Strategy 3: Fallback to first few paragraphs
                if not content:
                    paragraphs = soup.find_all("p")[:10]  # Get first 10 paragraphs
                    content = "\n".join(p.get_text().strip() for p in paragraphs)
                
                # Combine all content
                scraped_content = f"""
Title: {title}

Description: {description}

Keywords: {keywords}

Content:
{content}
"""
                return scraped_content.strip()
    except Exception as e:
        logger.error(f"Error in generic scraping for {url}: {str(e)}")
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

{'\n\n'.join(structured_content[:5])}  # Limit to first 5 sections for brevity
"""
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

# Main plugin class
class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.config.load_and_update()
        self.log.info("MatrixToDiscourseBot started")
        self.ai_integration = AIIntegration(self.config, self.log)
        self.discourse_api = DiscourseAPI(self.config, self.log)
        self.target_audience = self.config["target_audience"]

    # Function to get the configuration class
    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config
    # Command to handle the help event
    @command.new(name=lambda self: self.config["help_trigger"], require_subcommand=False)
    async def help_command(self, evt: MessageEvent) -> None:
        await self.handle_help(evt)

    async def handle_help(self, evt: MessageEvent) -> None:
        self.log.info(f"Command !{self.config['help_trigger']} triggered.")
        help_msg = (
            "Welcome to the Community Forum Bot!\n\n"
            f"To create a post on the forum, reply to a message with `!{self.config['post_trigger']}`.\n"
            f"To summarize the last N messages, use `!{self.config['post_trigger']} -n <number>`.\n"
            f"To summarize messages from a timeframe, use `!{self.config['post_trigger']} -h <hours> -m <minutes> -d <days>`.\n"
            f"To search the forum, use `!{self.config['search_trigger']} <query>`.\n"
            f"To post a URL, reply to a message containing a URL with `!{self.config['url_post_trigger']}`.\n"
            f"For help, use `!{self.config['help_trigger']}`."
        )
        await evt.reply(help_msg)

    # Command to handle the post command
    @command.new(name=lambda self: self.config["post_trigger"], require_subcommand=False)
    @command.argument("args", pass_raw=True, required=False)
    async def post_to_discourse_command(self, evt: MessageEvent, args: str = None) -> None:
        await self.handle_post_to_discourse(evt, args)

    async def handle_post_to_discourse(self, evt: MessageEvent, args: str = None) -> None:
        post_trigger = self.config["post_trigger"]
        self.log.info(f"Command !{post_trigger} triggered.")
        # Parse the arguments for the command
        parser = argparse.ArgumentParser(prog=f"!{post_trigger}", add_help=False)
        parser.add_argument("-n", "--number", type=int, help="Number of messages to summarize")
        parser.add_argument("-h", "--hours", type=int, help="Number of hours to look back")
        parser.add_argument("-m", "--minutes", type=int, help="Number of minutes to look back")
        parser.add_argument("-d", "--days", type=int, help="Number of days to look back")
        parser.add_argument("title", nargs='*', help="Optional title for the post")

        try:
            args_namespace = parser.parse_args(args.split() if args else [])
        except Exception as e:
            await evt.reply(f"Error parsing arguments: {e}")
            return

        # Extract values
        number = args_namespace.number
        hours = args_namespace.hours
        minutes = args_namespace.minutes
        days = args_namespace.days
        title = ' '.join(args_namespace.title) if args_namespace.title else None

        # Validate arguments
        if (number and (hours or minutes or days)):
            await evt.reply("Please specify either a number of messages (-n) or a timeframe (-h/-m/-d), not both.")
            return

        if not (number or hours or minutes or days or evt.content.get_reply_to()):
            await evt.reply("Please reply to a message or specify either a number of messages (-n) or a timeframe (-h/-m/-d).")
            return

        # Fetch messages and check if the number of messages is less than the number requested
        messages = []
        if number: # Fetch messages by number meaning the number of messages to summarize
            messages = await self.fetch_messages_by_number(evt, number)
            if len(messages) < number:
                await evt.reply(f"Only found {len(messages)} messages to summarize.")
        elif hours or minutes or days:
            total_seconds = (days or 0) * 86400 + (hours or 0) * 3600 + (minutes or 0) * 60
            time_delta = timedelta(seconds=total_seconds)
            messages = await self.fetch_messages_by_time(evt, time_delta)
            if not messages:
                await evt.reply("No messages found in the specified timeframe.")
                return
        else:
            # Default to the replied-to message
            replied_event = await evt.client.get_event(
                evt.room_id, evt.content.get_reply_to()
            )
            messages = [replied_event]

        if not messages:
            await evt.reply("No messages found to summarize.")
            return

        # Combine message bodies, including the replied-to message
        message_bodies = [event.content.body for event in reversed(messages) if hasattr(event.content, 'body')]
        combined_message = "\n\n".join(message_bodies)

        # Generate summary using AI model for multiple messages leave individuals out of summary command
        # Generate summary using AI model for multiple messages
        if messages:
            if len(messages) > 1:
                # Summarize the combined message content
                summary = await self.ai_integration.summarize_content(combined_message)
            else:
                # Use the single message directly as the summary
                #summarize the single message and add it to the content of the post above the original message
                head_summary = await self.ai_integration.summarize_content(message_bodies[0])
                if head_summary:
                    # Add the summary to the content of the post above the original message
                    summary = f"{head_summary}\n\n---\n\nOriginal Message:\n{message_bodies[0]}"
                else:
                    # If the summary fails, use the original message without the summary
                    summary = message_bodies[0]
        else:
            # Prompt the user if no messages are found to summarize
            await evt.reply("Please reply to a message to summarize.")
            return

        # Fallback to the combined message if summarization fails
        if not summary:
            self.log.warning("AI summarization failed. Falling back to the original message content.")
            summary = combined_message

        # Determine if a title is required
        ai_model_type = self.config["ai_model_type"]

        if ai_model_type == "none":
            # If AI is set to 'none', title is required from the user
            if not title:
                await evt.reply(
                    "A title is required since AI model is set to 'none'. Please provide a title."
                )
                return
        else:
            # Use provided title or generate one using the AI model
            if not title:
                title = await self.generate_title(summary)
            if not title:
                title = "Untitled Post"  # Fallback title if generation fails

        self.log.info(f"Generated Title: {title}")

        # Generate tags using AI model
        tags = await self.ai_integration.generate_tags(summary)
        if not tags:
            tags = ["bot-post"]  # Fallback tags if generation fails

        # Get the topic ID based on the room ID
        #function to get the topic ID based on the room ID based on the config
        if self.config["matrix_to_discourse_topic"]:
            category_id = self.config["matrix_to_discourse_topic"].get(evt.room_id)
        else:
            category_id = self.config["unsorted_category_id"]

        # Log the category ID being used
        self.log.info(f"Using category ID: {category_id}")

        # Create the post on Discourse
        post_url, error = await self.discourse_api.create_post(
            title=title,
            raw=summary,
            category_id=category_id,
            tags=tags,
        )
        if post_url:
            posted_link_url = f"{self.config['discourse_base_url']}/tag/posted-link"
            # post_url should not be markdown
            post_url = post_url.replace("[", "").replace("]", "")
            bypass_links = generate_bypass_links(url)  # Ensure bypass links are generated
            await evt.reply(
                f"🔗 {title}\n"
                f"**Forum Post URL:** {post_url}\n\n"
                f"**Summary Preview:** {summary[:self.config['summary_length_in_characters']]}...\n\n"
            )
        else:
            await evt.reply(f"Failed to create post: {error}")

    async def fetch_messages_by_number(self, evt: MessageEvent, number: int) -> List[MessageEvent]:
        """
        Fetch up to `number` messages from this room (excluding the bot's own messages).
        Returns a list of MessageEvent objects from newest to oldest.
        """
        messages = []
        last_token: Optional[str] = None
        limit = 100  # how many events to fetch per call

        self.log.info(f"Fetching last {number} messages in room {evt.room_id}...")

        while len(messages) < number:
            try:
                chunk = await self.client.get_room_messages(
                    room_id=evt.room_id,
                    from_token=last_token,
                    direction=PaginationDirection.BACKWARD,
                    limit=limit
                )
            except Exception as e:
                self.log.error(f"Error in get_room_messages: {e}")
                break

            if not chunk.chunk:
                break

            for event in chunk.chunk:
                if event.type == EventType.ROOM_MESSAGE and event.sender != self.client.mxid:
                    messages.append(event)
                    if len(messages) >= number:
                        break

            if len(chunk.chunk) < limit or not chunk.end:
                break

            last_token = chunk.end

        return messages[:number]

    async def fetch_messages_by_time(self, evt: MessageEvent, time_delta: timedelta) -> List[MessageEvent]:
        """
        Fetch all messages sent within the last `time_delta` (excluding bot's own).
        Returns messages from newest to oldest that fall within the timeframe.
        """
        messages = []
        last_token: Optional[str] = None
        limit = 100
        cutoff_ms = (datetime.now(tz=timezone.utc) - time_delta).timestamp() * 1000

        self.log.info(f"Fetching messages in the last {time_delta} from room {evt.room_id}...")

        while True:
            try:
                chunk = await self.client.get_room_messages(
                    room_id=evt.room_id,
                    from_token=last_token,
                    direction=PaginationDirection.BACKWARD,
                    limit=limit
                )
            except Exception as e:
                self.log.error(f"Error in get_room_messages: {e}")
                break

            if not chunk.chunk:
                break

            for event in chunk.chunk:
                if event.type == EventType.ROOM_MESSAGE and event.sender != self.client.mxid:
                    if event.server_timestamp < cutoff_ms:
                        return messages
                    messages.append(event)

            if len(chunk.chunk) < limit or not chunk.end:
                break

            last_token = chunk.end

        return messages

    async def summarize_messages_as_post(self, evt: MessageEvent,
                                         number: Optional[int] = None,
                                         hours: Optional[int] = None,
                                         minutes: Optional[int] = None,
                                         days: Optional[int] = None) -> Optional[str]:
        """
        Summarize either the last N messages or messages from a timeframe.
        If no arguments are provided, fallback to the replied-to message.
        """
        if (number and (hours or minutes or days)):
            await evt.reply("Please specify either a number of messages (-n) or a timeframe (-h/-m/-d), not both.")
            return None
        if not (number or hours or minutes or days or evt.content.get_reply_to()):
            await evt.reply("Please reply to a message or specify a number of messages (-n) or timeframe (-h/-m/-d).")
            return None

        messages: List[MessageEvent] = []
        if number:
            messages = await self.fetch_messages_by_number(evt, number)
            if not messages:
                await evt.reply("No messages found for that count.")
                return None
            if len(messages) < number:
                await evt.reply(f"Only found {len(messages)} messages to summarize.")
        elif (hours or minutes or days):
            total_seconds = (days or 0) * 86400 + (hours or 0) * 3600 + (minutes or 0) * 60
            time_delta = timedelta(seconds=total_seconds)
            messages = await self.fetch_messages_by_time(evt, time_delta)
            if not messages:
                await evt.reply("No messages found in the specified timeframe.")
                return None
        else:
            replied_id = evt.content.get_reply_to()
            if not replied_id:
                await evt.reply("No messages found to summarize.")
                return None
            replied_event = await evt.client.get_event(evt.room_id, replied_id)
            messages = [replied_event] if replied_event else []

        if not messages:
            await evt.reply("No messages found to summarize.")
            return None

        messages.reverse()
        combined_text = "\n\n".join(
            evt.content.body for evt in messages if hasattr(evt.content, 'body')
        )

        summary = await self.ai_integration.summarize_content(combined_text)
        if not summary:
            self.log.warning("AI summarization returned empty or failed; falling back to raw text.")
            summary = combined_text

        return summary

    # Function to generate title for bypassed links
    async def generate_title_for_bypassed_links(self, message_body: str) -> Optional[str]:
        # Remove the use_links_prompt parameter since generate_links_title doesn't accept it
        return await self.ai_integration.generate_links_title(message_body)
    
    # Function to generate title
    async def generate_title(self, message_body: str) -> Optional[str]:
        return await self.ai_integration.generate_title(message_body, use_links_prompt=False)

    # Command to search the discourse
    @command.new(name=lambda self: self.config["search_trigger"], require_subcommand=False)
    @command.argument("query", pass_raw=True, required=True)
    async def search_discourse_command(self, evt: MessageEvent, query: str) -> None:
        await self.handle_search_discourse(evt, query)

    async def handle_search_discourse(self, evt: MessageEvent, query: str) -> None:
        self.log.info(f"Command !{self.config['search_trigger']} triggered.")
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
            self.log.error(f"Error processing !{self.config['search_trigger']} command: {e}")
            await evt.reply(f"An error occurred: {e}")

    # Handle messages with URLs and process them if url_listening is enabled.
    @event.on(EventType.ROOM_MESSAGE)
    async def handle_message(self, evt: MessageEvent) -> None:
        # If URL listening is disabled, don't process URL patterns.
        if not self.config.get("url_listening", False):
            return

        if evt.sender == self.client.mxid:
            return

        if evt.content.msgtype != MessageType.TEXT:
            return

        # Extract URLs from the message body.
        message_body = evt.content.body
        urls = extract_urls(message_body)
        if urls:
            await self.process_link(evt, message_body)

    # Command to process URLs in replies
    @command.new(name=lambda self: self.config["url_post_trigger"], require_subcommand=False)
    async def post_url_to_discourse_command(self, evt: MessageEvent) -> None:
        await self.handle_post_url_to_discourse(evt)

    async def handle_post_url_to_discourse(self, evt: MessageEvent) -> None:
        self.log.info(f"Command !{self.config['url_post_trigger']} triggered.")

        if not evt.content.get_reply_to():
            await evt.reply("You must reply to a message containing a URL to use this command.")
            return

        replied_event = await evt.client.get_event(
            evt.room_id, evt.content.get_reply_to()
        )
        message_body = replied_event.content.body

        if not message_body:
            await evt.reply("The replied-to message is empty.")
            return

        urls = extract_urls(message_body)
        if not urls:
            await evt.reply("No URLs found in the replied message.")
            return

        await self.process_link(evt, message_body)
        
    async def process_link(self, evt: MessageEvent, message_body: str) -> None:
        urls = extract_urls(message_body)
        username = evt.sender.split(":")[0]  # Extract the username from the sender
        
        # Filter URLs based on patterns and blacklist
        urls_to_process = [url for url in urls if self.config.should_process_url(url)]

        if not urls_to_process:
            self.log.debug("No URLs matched the configured patterns or all URLs were blacklisted")
            return

        for url in urls_to_process:
            # Check for duplicates
            duplicate_exists, duplicate_url = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"This URL has already been posted, a post summarizing and linking to 12ft.io and archive.org is already on the forum here: {duplicate_url}")
                continue

            # Scrape content
            content = await scrape_content(url)
            if content:
                # Summarize content if scraping was successful
                summary = await self.ai_integration.summarize_content(content, user_message=message_body)
                if not summary:
                    self.log.warning(f"Summarization failed for URL: {url}")
                    summary = "Content could not be scraped or summarized."
            else:
                self.log.warning(f"Scraping content failed for URL: {url}")
                summary = "Content could not be scraped or summarized. The original site may have access restrictions."

            # Generate title
            title = await self.generate_title_for_bypassed_links(message_body)
            if not title:
                self.log.info(f"Generating title using URL and domain for: {url}")
                title = await self.generate_title_for_bypassed_links(f"URL: {url}, Domain: {url.split('/')[2]}")
                if not title:
                    title = f"Link from {url.split('/')[2]}"

            # Generate tags
            tags = await self.ai_integration.generate_tags(content)
            # Log the generated tags for debugging
            self.log.debug(f"Generated tags: {tags}")
            
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
                    self.log.debug(f"Archive link {name} is not working for {url}")
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
                f"{summary}\n\n"
                f"{archive_links_section}\n"
                f"**Original Link:** <{url}>\n\n"
                f"User Message: {message_body}\n\n"
                f"For more on bypassing paywalls, see the [post on bypassing methods](https://forum.irregularchat.com/t/bypass-links-and-methods/98?u=sac)"
            )

            # Determine category ID based on room ID
            category_id = self.config["matrix_to_discourse_topic"].get(evt.room_id, self.config["unsorted_category_id"])

            # Log the category ID being used
            self.log.info(f"Using category ID: {category_id}")
            self.log.info(f"Final tags being used: {tags}")

            # Create the post on Discourse
            post_url, error = await self.discourse_api.create_post(
                title=title,
                raw=post_body,
                category_id=category_id,
                tags=tags,
            )

            if post_url:
                posted_link_url = f"{self.config['discourse_base_url']}/tag/posted-link"
                # post_url should not be markdown
                post_url = post_url.replace("[", "").replace("]", "")
                await evt.reply(
                    f"🔗 {title}\n\n"
                    f"**Forum Post URL:** {post_url}\n\n"
                    f"{summary[:self.config['summary_length_in_characters']]}...\n\n"
                )
            else:
                await evt.reply(f"Failed to create post: {error}")

    async def search_mediawiki(self, query: str) -> Optional[List[Dict]]:
        """
        Search the MediaWiki API using the configured base URL.
        Tries both /w/api.php and /api.php, returning a list of search result dictionaries or None on failure.
        """
        mediawiki_base_url = self.config.get("mediawiki_base_url", None)
        if not mediawiki_base_url:
            self.log.error("MediaWiki base URL is not configured.")
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
            self.log.debug(f"Trying MediaWiki API candidate URL: {candidate_url} with params: {params}")
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(candidate_url, params=params) as response:
                        if response.status == 200:
                            resp_json = await response.json()
                            self.log.debug(f"Candidate URL {candidate_url} returned: {resp_json}")
                            # Check if the response has the expected structure.
                            if "query" in resp_json and "search" in resp_json["query"]:
                                return resp_json["query"]["search"]
                            else:
                                self.log.debug(f"Candidate URL {candidate_url} did not return expected keys: {resp_json}")
                        else:
                            body = await response.text()
                            self.log.debug(f"Candidate URL {candidate_url} failed with status: {response.status} and body: {body}")
            except Exception as e:
                self.log.error(f"Error searching MediaWiki at candidate URL {candidate_url}: {e}")
        
        self.log.error("All candidate URLs failed for MediaWiki search.")
        return None
