# bot.py

import asyncio
import json
import re
from datetime import datetime
from typing import Type, Dict, List
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
import aiohttp
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper
from url_handler import extract_urls, generate_bypass_links, scrape_content
from ai_integration import AIIntegration
from discourse_api import DiscourseAPI


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

        # URL patterns to check for in the message body
        helper.copy("url_patterns", is_list=True)


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

    # Function to handle the help event
    @command.new(name=lambda self: self.config["help_trigger"], require_subcommand=False)
    async def help(self, evt: MessageEvent) -> None:
        self.log.info(f"Command !{self.config['help_trigger']} triggered.")
        help_msg = (
            "Welcome to the Community Forum Bot!\n\n"
            f"To create a post on the forum, reply to a message with `!{self.config['post_trigger']}`.\n"
            f"To search the forum, use `!{self.config['search_trigger']} <query>`.\n"
            f"To post a URL, reply to a message containing a URL with `!{self.config['url_post_trigger']}`.\n"
            f"For help, use `!{self.config['help_trigger']}`."
        )
        await evt.reply(help_msg)

    # Function to handle the post command
    @command.new(name=lambda self: self.config["post_trigger"], require_subcommand=False)
    @command.argument("title", pass_raw=True, required=False)
    async def post_to_discourse(self, evt: MessageEvent, title: str = None) -> None:
        self.log.info(f"Command !{self.config['post_trigger']} triggered.")
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
                    title = "Default Title"  # Fallback title if generation fails

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
            self.log.error(f"Error processing !{self.config['post_trigger']} command: {e}")
            await evt.reply(f"An error occurred: {e}")

    # Function to generate title
    async def generate_title(self, message_body: str) -> str:
        return await self.ai_integration.generate_title(message_body)

    # Function to search the discourse
    @command.new(name=lambda self: self.config["search_trigger"], require_subcommand=False)
    @command.argument("query", pass_raw=True, required=True)
    async def search_discourse(self, evt: MessageEvent, query: str) -> None:
        self.log.info(f"Command !{self.config['search_trigger']} triggered.")
        await evt.reply("Searching the forum...")

        try:
            search_results = await self.discourse_api.search_discourse(query)
            if search_results is not None:
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

                if search_results:
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
        if evt.content.msgtype != MessageType.TEXT:
            return

        message_body = evt.content.body
        url_patterns = self.config["url_patterns"]
        for pattern in url_patterns:
            if re.search(pattern, message_body):
                await self.process_link(evt, message_body)
                break

    # Command to process URLs in replies
    @command.new(name=lambda self: self.config["url_post_trigger"], require_subcommand=False)
    async def post_url_to_discourse(self, evt: MessageEvent) -> None:
        self.log.info(f"Command !{self.config['url_post_trigger']} triggered.")

        if not evt.content.get_reply_to():
            await evt.reply("You must reply to a message containing a URL to use this command.")
            return

        replied_event = await evt.client.get_event(
            evt.room_id, evt.content.get_reply_to()
        )
        message_body = replied_event.content.body

        urls = extract_urls(message_body)
        if not urls:
            await evt.reply("No URLs found in the replied message.")
            return

        await self.process_link(evt, message_body)

    # Process links in messages
    async def process_link(self, evt: MessageEvent, message_body: str) -> None:
        urls = extract_urls(message_body)
        for url in urls:
            # Check for duplicates
            duplicate_exists = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"A post with this URL already exists: {url}")
                continue

            # Scrape content
            content = await scrape_content(url)
            if not content:
                await evt.reply(f"Failed to scrape content from {url}")
                continue

            # Summarize content
            summary = await self.ai_integration.summarize_content(content)
            if not summary:
                await evt.reply("Failed to summarize the content.")
                continue

            # Generate bypass links
            bypass_links = generate_bypass_links(url)

            # Prepare message body
            message_body = (
                f"{summary}\n\n"
                f"Original Link: {bypass_links['original']}\n"
                f"12ft.io Link: {bypass_links['12ft']}\n"
                f"Archive.org Link: {bypass_links['archive']}"
            )

            # Generate title
            title = await self.generate_title(summary)
            if not title:
                title = "Default Title"

            # Create the post on Discourse
            tags = ["posted-link"]
            topic_id = self.config["unsorted_category_id"]
            post_url, error = await self.discourse_api.create_post(
                title=title,
                raw=message_body,
                category_id=topic_id,
                tags=tags,
            )

            if post_url:
                await evt.reply(f"Post created successfully! URL: {post_url}")
            else:
                await evt.reply(f"Failed to create post: {error}")