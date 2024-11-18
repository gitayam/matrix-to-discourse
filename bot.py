import asyncio
import json
import re
import traceback
import aiohttp
import logging
import argparse
from datetime import datetime, timedelta
from typing import Type, Dict, List, Optional
from mautrix.types import Direction
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

from html.parser import HTMLParser

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

# AIIntegration class
class AIIntegration:
    def __init__(self, config, log):
        self.config = config
        self.log = log

    async def generate_title(self, message_body: str) -> Optional[str]:
        ai_model_type = self.config["ai_model_type"]

        if ai_model_type == "openai":
            return await self.generate_openai_title(message_body)
        elif ai_model_type == "local":
            return await self.generate_local_title(message_body)
        elif ai_model_type == "google":
            return await self.generate_google_title(message_body)
        else:
            self.log.error(f"Unknown AI model type: {ai_model_type}")
            return None

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

    async def summarize_with_local_llm(self, content: str) -> Optional[str]:
        # Implement according to local LLM API
        self.log.error("Local LLM integration is not implemented yet.")
        return None

    async def generate_google_title(self, message_body: str) -> Optional[str]:
        prompt = f"Create a brief (3-10 word) attention-grabbing title for the following post on the community forum: {message_body}"
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
        prompt = f"Please provide a concise summary of the following content:\n\n{content}"
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

    async def check_for_duplicate(self, url: str) -> bool:
        search_url = f"{self.config['discourse_base_url']}/search.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }
        params = {"q": url}

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

def generate_bypass_links(url: str) -> Dict[str, str]:
    links = {
        "original": url,
        "12ft": f"https://12ft.io/{url}",
        "archive": f"https://web.archive.org/web/{url}",
    }
    return links

async def scrape_content(url: str) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve content from {url}: HTTP {response.status}")
                    return None
                html = await response.text()
                cleaner = HTMLCleaner()
                cleaner.feed(html)
                content = cleaner.get_cleaned_text()
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
    @command.new(name=lambda self: self.config["post_trigger"], require_subcommand=False, pass_raw=True)
    async def post_to_discourse_command(self, evt: MessageEvent, args: str = None) -> None:
        await self.handle_post_to_discourse(evt, args)

    async def handle_post_to_discourse(self, evt: MessageEvent, args: str = None) -> None:
        post_trigger = self.config["post_trigger"]
        self.log.info(f"Command !{post_trigger} triggered.")

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

        # Fetch messages
        messages = []
        if number:
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

        # Combine message bodies
        message_bodies = [event.content.body for event in reversed(messages) if hasattr(event.content, 'body')]
        combined_message = "\n\n".join(message_bodies)

        # Generate summary using AI model
        summary = await self.ai_integration.summarize_content(combined_message)
        if not summary:
            self.log.warning("Failed to generate summary.")
            summary = combined_message  # Fallback to combined message

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

        # Get the topic ID based on the room ID
        topic_id = self.config["matrix_to_discourse_topic"].get(
            evt.room_id, self.config["unsorted_category_id"]
        )

        # Create the post on Discourse
        tags = []  # Modify tags as needed
        post_url, error = await self.discourse_api.create_post(
            title=title,
            raw=summary,
            category_id=topic_id,
            tags=tags,
        )
        if post_url:
            await evt.reply(
                f"Post created successfully! Title: {title}, URL: {post_url} \n\n Log in to the community to engage with this post."
            )
        else:
            await evt.reply(f"Failed to create post: {error}")

    # Fetch messages by number
    async def fetch_messages_by_number(self, evt: MessageEvent, number: int) -> List[MessageEvent]:
        messages = []
        prev_batch = evt.event_id
        while len(messages) < number:
            response = await self.client.get_messages(evt.room_id, prev_batch, direction=Direction.BACKWARD, limit=100)
            chunk = response.chunk
            if not chunk:
                break  # No more messages
            for event in chunk:
                if event.type == EventType.ROOM_MESSAGE and event.sender != self.client.mxid:
                    messages.append(event)
                    if len(messages) >= number:
                        break
            prev_batch = response.end
        return messages[:number]
    # Fetch messages by timeframe
    async def fetch_messages_by_time(self, evt: MessageEvent, time_delta: timedelta) -> List[MessageEvent]:
        messages = []
        prev_batch = evt.event_id
        end_time = datetime.utcnow() - time_delta

        while True:
            response = await self.client.get_messages(evt.room_id, prev_batch, direction=Direction.BACKWARD, limit=100)
            chunk = response.chunk
            if not chunk:
                break  # No more messages
            for event in chunk:
                if event.type == EventType.ROOM_MESSAGE and event.sender != self.client.mxid:
                    event_time = datetime.utcfromtimestamp(event.server_timestamp / 1000)
                    if event_time < end_time:
                        return messages
                    messages.append(event)
            prev_batch = response.end
        return messages
    # Function to generate title
    async def generate_title(self, message_body: str) -> Optional[str]:
        return await self.ai_integration.generate_title(message_body)

    # Command to search the discourse
    @command.new(name=lambda self: self.config["search_trigger"], require_subcommand=False)
    @command.argument("query", pass_raw=True, required=True)
    async def search_discourse_command(self, evt: MessageEvent, query: str) -> None:
        await self.handle_search_discourse(evt, query)

    async def handle_search_discourse(self, evt: MessageEvent, query: str) -> None:
        self.log.info(f"Command !{self.config['search_trigger']} triggered.")
        await evt.reply("Searching the forum...")

        try:
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

                    def format_results(results):
                        return "\n".join(
                            [
                                f"* [{result['title']}]({self.config['discourse_base_url']}/t/{result['slug']}/{result['id']})"
                                for result in results
                            ]
                        )

                    result_msg = (
                        "**Top 2 Most Recent:**\n"
                        + format_results(top_recent)
                        + "\n\n**Top 3 Most Seen:**\n"
                        + format_results(top_seen)
                    )

                    await evt.reply(f"Search results:\n{result_msg}")
                else:
                    await evt.reply("No results found.")
            else:
                await evt.reply("Failed to perform search.")
        except Exception as e:
            self.log.error(f"Error processing !{self.config['search_trigger']} command: {e}")
            await evt.reply(f"An error occurred: {e}")

    # Handle messages with URLs
    @event.on(EventType.ROOM_MESSAGE)
    async def handle_message(self, evt: MessageEvent) -> None:
        if evt.sender == self.client.mxid:
            return

        if evt.content.msgtype != MessageType.TEXT:
            return
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

    # Process links in messages
    async def process_link(self, evt: MessageEvent, message_body: str) -> None:
        urls = extract_urls(message_body)
        username = evt.sender.split(":")[0]  # Extract the username from the sender
        for url in urls:
            # Check for duplicates
            duplicate_exists = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"A post with this URL already exists: {url}")
                continue

            # Scrape content
            content = await scrape_content(url)
            summary = None
            if content:
                # Summarize content if scraping was successful
                summary = await self.ai_integration.summarize_content(content)
                if not summary:
                    self.log.warning(f"Summarization failed for URL: {url}")
            else:
                self.log.warning(f"Scraping content failed for URL: {url}")

            # Generate title
            title = None
            if summary:
                title = await self.generate_title(summary)
            else:
                self.log.info(f"Generating title using URL and domain for: {url}")
                title = await self.generate_title(f"URL: {url}, Domain: {url.split('/')[2]}")

            if not title:
                title = "Untitled Post"

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
                # Include title in the reply
                await evt.reply(f"Post created successfully! Title: {title}, URL: {post_url}")
            else:
                await evt.reply(f"Failed to create post: {error}")