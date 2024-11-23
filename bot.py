import asyncio
import json
import re
import traceback
import aiohttp
import logging
from datetime import datetime
from typing import Type, Dict, List, Optional

from mautrix.client import Client
from mautrix.types import (
    Format,
    TextMessageEventContent,
    EventType,
    RelationType,
    MessageType,
)
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper
# from bs4 import BeautifulSoup

from html.parser import HTMLParser

# Define HTMLCleaner to extract text from HTML
class HTMLCleaner(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_cleaned_text(self):
        return ''.join(self.text).strip()

# Configure logging
logger = logging.getLogger(__name__)

# Config class to manage configuration
class Config(BaseProxyConfig):
    #consider validating types or defaulting missing values to avoid runtime errors.
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

        # Handle URL patterns explicitly as a list
        if "url_patterns" in helper.base:
            self["url_patterns"] = list(helper.base["url_patterns"])
        else:
            self["url_patterns"] = []

        # configure for default title fallback if AI integration fails
        helper.copy("default_title")        

# AIIntegration class, handles the AI integration
class AIIntegration:
    def __init__(self, config, log):
        self.config = config
        self.log = log
    # Generate the title based on the AI model type
    async def generate_title(self, message_body: str) -> Optional[str]:
        ai_model_type = self.config["ai_model_type"]
        # Check the AI model type and call the appropriate method
        if ai_model_type == "openai":
            return await self.generate_openai_title(message_body)
        elif ai_model_type == "local":
            return await self.generate_local_title(message_body)
        elif ai_model_type == "google":
            return await self.generate_google_title(message_body)
        else:
            self.log.error(f"Unknown AI model type: {ai_model_type}")
            return None
    # Summarize the content based on the AI model type
    async def summarize_content(self, content: str) -> Optional[str]:
        ai_model_type = self.config["ai_model_type"]
        if ai_model_type == "openai":
            return await self.summarize_with_openai(content)
        elif ai_model_type == "local":
            return await self.summarize_with_local_llm(content)
        elif ai_model_type == "google":
            return await self.summarize_with_google(content)
        else:
            self.log.error(f"Unknown AI model type: {ai_model_type}")
            return None

    # Implement the methods for each AI model
    async def generate_openai_title(self, message_body: str) -> Optional[str]:
        prompt = f"Create a brief (3-10 word) attention-grabbing title for the following post on the community forum: {message_body}"
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
            # Make the request to the OpenAI API
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
    # Summarize the content with the OpenAI API
    async def summarize_with_openai(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary of the following content:\n\n{content}"
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
            # Make the request to the OpenAI API
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

    async def generate_local_title(self, message_body: str) -> Optional[str]:
        # Implement according to local LLM API
        self.log.error("Local LLM integration is not implemented yet.")
        return None
    # Summarize the content with the local LLM API
    async def summarize_with_local_llm(self, content: str) -> Optional[str]:
        # Implement according to local LLM API
        self.log.error("Local LLM integration is not implemented yet.")
        return None
    # Generate the title with the Google API
    async def generate_google_title(self, message_body: str) -> Optional[str]:
        # Implement according to Google API
        self.log.error("Google AI integration is not implemented yet.")
        return None
    # Summarize the content with the Google API
    async def summarize_with_google(self, content: str) -> Optional[str]:
        # Implement according to Google API
        self.log.error("Google AI integration is not implemented yet.")
        return None

# DiscourseAPI class
class DiscourseAPI:
    def __init__(self, config, log):
        self.config = config
        self.log = log
    # Create the post on Discourse
    async def create_post(self, title, raw, category_id, tags=None):
        url = f"{self.config['discourse_base_url']}/posts.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        payload = {
            "title": title,
            "raw": raw,
            "category": category_id,
            "tags": tags or [],
        }
        # Make the request to the Discourse API for creating the post
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
    # Check for duplicates URL on Discourse
    async def check_for_duplicate(self, url: str) -> bool:
        search_url = f"{self.config['discourse_base_url']}/search.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        params = {"q": url}
        # Make the request to the Discourse API for checking for duplicates
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, params=params) as response:
                response_text = await response.text()
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError as e:
                    self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                    return False

                if response.status == 200:
                    return bool(response_json.get("topics"))
                else:
                    self.log.error(f"Discourse API error: {response.status} {response_json}")
                    return False
    # Search the Discourse for a query
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

# URL handling functions
def extract_urls(text: str) -> List[str]:
    url_regex = r'(https?://\S+)'
    return re.findall(url_regex, text)
# Generate bypass links based on the URL
def generate_bypass_links(url: str) -> Dict[str, str]:
    links = {
        "original": url,
        "12ft": f"https://12ft.io/{url}",
        "archive": f"https://web.archive.org/web/{url}",
    }
    return links

# Function to scrape content from URLs
#FIXME: Get BeautifulSoup to work and focus on gaining the title and keywords and summary from the html
#to pass to the AI model for generating the title and summary
async def scrape_content(url: str) -> Optional[str]:
    try:
        # Use beautifulsoup to parse the html
        async with aiohttp.ClientSession() as session:
            # Get the response from the url
            async with session.get(url, timeout=10) as response:
                # Check if the response is ok
                if response.status != 200:
                    # Log the error
                    logger.error(f"Failed to retrieve content from {url}: HTTP {response.status}")
                    return None
                # Get the html from the response
                html = await response.text()
                # Parse the html
                soup = HTMLCleaner()
                # Get the Title and keywords from the html
                title = soup.title.string if soup.title else None
                keywords = [meta.get('content') for meta in soup.find_all('meta') if meta.get('name') == 'keywords']    
                # Get the text from the html
                content = soup.get_text()
                # Return the content if it exists
                return content if content else None
    except Exception as e:
        logger.error(f"Error scraping content from {url}: {e}", exc_info=True)
        return None

# Main plugin class
class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.config.load_and_update()
        self.log.info("MatrixToDiscourseBot started")
        self.ai_integration = AIIntegration(self.config, self.log)
        self.discourse_api = DiscourseAPI(self.config, self.log)

    # Function to get the configuration class
    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    # Command to handle the help event
    @command.new(name="help", require_subcommand=False)
    async def help_command(self, evt: MessageEvent) -> None:
        await self.handle_help(evt)
    # Handle the help event
    async def handle_help(self, evt: MessageEvent) -> None:
        help_trigger = self.config["help_trigger"]
        post_trigger = self.config["post_trigger"]
        search_trigger = self.config["search_trigger"]
        url_post_trigger = self.config["url_post_trigger"]
        # Log the help trigger
        self.log.info(f"Command !{help_trigger} triggered.")
        help_msg = (
            "Welcome to the Community Forum Bot!\n\n"
            f"To create a post on the forum, reply to a message with `!{post_trigger}`.\n"
            f"To search the forum, use `!{search_trigger} <query>`.\n"
            f"To post a URL, reply to a message containing a URL with `!{url_post_trigger}`.\n"
            f"For help, use `!{help_trigger}`."
        )
        await evt.reply(help_msg)

    # Command to handle the post command
    @command.new(name="post", require_subcommand=False)
    @command.argument("title", pass_raw=True, required=False)
    async def post_to_discourse_command(self, evt: MessageEvent, title: str = None) -> None:
        await self.handle_post_to_discourse(evt, title)

    async def handle_post_to_discourse(self, evt: MessageEvent, title: str = None) -> None:
        post_trigger = self.config["post_trigger"]
        self.log.info(f"Command !{post_trigger} triggered.")
        await evt.reply(
            "Creating a Forum post, log in to the community forum to view all posts and to engage on the forum..."
        )

        try:
            self.log.info(f"Received event body: {evt.content.body}")

            # Check if the message is a reply to another message
            if not evt.content.get_reply_to():
                await evt.reply("You must reply to a message to use this command.")
                return

            # Extract the body of the replied-to message
            replied_event = await evt.client.get_event(
                evt.room_id, evt.content.get_reply_to()
            )
            message_body = replied_event.content.body
            self.log.info(f"Message body: {message_body}")

            if not message_body:
                await evt.reply("The replied-to message is empty.")
                return

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
                    title = await self.generate_title(message_body)
                if not title:
                    title = default_title  # Fallback title if generation fails

            self.log.info(f"Generated Title: {title}")

            # Get the topic ID based on the room ID
            topic_id = self.config["matrix_to_discourse_topic"].get(
                evt.room_id, self.config["unsorted_category_id"]
            )

            # Create the post on Discourse
            tags = []  # You can modify tags as needed
            post_url, error = await self.discourse_api.create_post(
                title=title,
                raw=message_body,
                category_id=topic_id,
                tags=tags,
            )
            if post_url:
                await evt.reply(
                    f"Post created successfully! URL: {post_url} \n\n Log in to the community to engage with this post."
                )
            else:
                await evt.reply(f"Failed to create post: {error}")

        except Exception as e:
            self.log.error(f"Error processing !{post_trigger} command: {e}")
            await evt.reply(f"An error occurred: {e}")

    # Function to generate title
    async def generate_title(self, message_body: str) -> Optional[str]:
        return await self.ai_integration.generate_title(message_body)

    # Command to search the discourse
    @command.new(name="search", require_subcommand=False)
    @command.argument("query", pass_raw=True, required=True)
    async def search_discourse_command(self, evt: MessageEvent, query: str) -> None:
        await self.handle_search_discourse(evt, query)

    async def handle_search_discourse(self, evt: MessageEvent, query: str) -> None:
        search_trigger = self.config["search_trigger"]
        self.log.info(f"Command !{search_trigger} triggered.")
        await evt.reply("Searching the forum...")
        # Try to search the Discourse for the query and display the results
        try:
            # Make the request to the Discourse API for searching
            search_results = await self.discourse_api.search_discourse(query)
            if search_results is not None:
                if search_results:
                    # Process and display search results
                    # Safely get keys with default values
                    for result in search_results:
                        result["views"] = result.get("views", 0)
                        result["created_at"] = result.get("created_at", "1970-01-01T00:00:00Z")

                    # Sort search results by created_at for most recent and by views for most seen
                    sorted_by_recent = sorted(
                        search_results, key=lambda x: x["created_at"], reverse=True
                    )
                    sorted_by_views = sorted(search_results, key=lambda x: x["views"], reverse=True)

                    # Select top 2 most recent and top 3 most seen
                    top_recent = sorted_by_recent[:2]
                    top_seen = sorted_by_views[:3]

                    # Format the results
                    def format_results(results):
                        return "\n".join(
                            [
                                f"* [{result['title']}]({self.config['discourse_base_url']}/t/{result['slug']}/{result['id']})"
                                for result in results
                            ]
                        )
                    # Format the results message
                    result_msg = (
                        "**Top 2 Most Recent:**\n"
                        + format_results(top_recent)
                        + "\n\n**Top 3 Most Seen:**\n"
                        + format_results(top_seen)
                    )
                    # Send the results message
                    await evt.reply(f"Search results:\n{result_msg}")
                else:
                    await evt.reply("No results found.")
            else:
                await evt.reply("Failed to perform search.")
        except Exception as e:
            self.log.error(f"Error processing !{search_trigger} command: {e}")
            await evt.reply(f"An error occurred: {e}")

    # Handle messages with URLs
    @event.on(EventType.ROOM_MESSAGE)
    async def handle_message(self, evt: MessageEvent) -> None:
        # Ignore messages sent by the bot itself
        if evt.sender == self.client.mxid:
            return

        if evt.content.msgtype != MessageType.TEXT:
            return
        pass

    # Command to process URLs in replies
    @command.new(name="url", require_subcommand=False)
    async def post_url_to_discourse_command(self, evt: MessageEvent) -> None:
        await self.handle_post_url_to_discourse(evt)
    # Handle the post URL to Discourse command
    async def handle_post_url_to_discourse(self, evt: MessageEvent) -> None:
        url_post_trigger = self.config["url_post_trigger"]
        self.log.info(f"Command !{url_post_trigger} triggered.")
        # Check if the message is a reply to another message
        if not evt.content.get_reply_to():
            await evt.reply("You must reply to a message containing a URL to use this command.")
            return
        # Extract the body of the replied-to message
        replied_event = await evt.client.get_event(
            evt.room_id, evt.content.get_reply_to()
        )
        message_body = replied_event.content.body
        # Check if the replied-to message is empty
        if not message_body:
            await evt.reply("The replied-to message is empty.")
            return
        # Extract the URLs from the replied-to message
        urls = extract_urls(message_body)
        # Check if no URLs were found in the replied-to message
        if not urls:
            await evt.reply("No URLs found in the replied message.")
            return
        # Process the links in the replied-to message
        await self.process_link(evt, message_body)

    # Process links in messages and send the post to Discourse and reply to the message
    async def process_link(self, evt: MessageEvent, message_body: str) -> None:
        urls = extract_urls(message_body)
        username = evt.sender.split(":")[0]  # Extract the username from the sender
        try:
            displayname = await self.client.get_displayname(evt.sender) or username
        except Exception as e:
            self.log.error(f"Failed to get display name for {evt.sender}: {e}")
            displayname = username  # Fallback to username
        for url in urls:
            # Check for duplicates
            duplicate_exists = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"A post with this URL already exists: {url}")
                continue

            # Idempotency check: Ensure the post is not already being processed
            if self.is_processing(url):
                self.log.info(f"URL {url} is already being processed.")
                continue
            self.mark_as_processing(url)

            try:
                # Scrape content
                content = await scrape_content(url)
                summary = None
                if content:
                    summary = await self.ai_integration.summarize_content(content)
                    if not summary:
                        self.log.warning(f"Summarization failed for URL: {url}")
                else:
                    self.log.warning(f"Scraping content failed for URL: {url}")

                # Generate title
                title = await self.generate_title(summary or f"URL: {url}, Domain: {url.split('/')[2]}")
                if not title:
                    title = "Untitled Post by " + displayname

                # Generate bypass links
                bypass_links = generate_bypass_links(url)

                # Prepare message body
                post_body = (
                    f"**Posted by:** @{username}\n\n"
                    f"{summary or 'Content could not be scraped or summarized.'}\n\n"
                    f"**Original Link:** {bypass_links['original']}\n"
                    f"**12ft.io Link:** {bypass_links['12ft']}\n"
                    f"**Archive.org Link:** {bypass_links['archive']}"
                )

                # Create the post on Discourse
                tags = ["posted-link"]
                topic_id = self.config["unsorted_category_id"]
                post_url, error = await self.discourse_api.create_post(
                    title=title,
                    raw=post_body,
                    category_id=topic_id,
                    tags=tags,
                )

                if post_url:
                    await evt.reply(f"Post created successfully! Title: {title}, URL: {post_url}")
                else:
                    await evt.reply(f"Failed to create post: {error}")
            finally:
                self.unmark_as_processing(url)