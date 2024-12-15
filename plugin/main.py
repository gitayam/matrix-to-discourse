# main.py
import asyncio
import json
import re
import traceback
import aiohttp
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Type, Dict, List, Optional

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

from plugin.config import Config
from plugin.ai_integration import AIIntegration
from plugin.discourse_api import DiscourseAPI
from plugin.utils import extract_urls, generate_bypass_links, scrape_content

logger = logging.getLogger(__name__)

class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.config.load_and_update()
        self.log.info("MatrixToDiscourseBot started")
        self.ai_integration = AIIntegration(self.config, self.log)
        self.discourse_api = DiscourseAPI(self.config, self.log)
        self.target_audience = self.config["target_audience"]

    @classmethod
    def get_config_class(cls) -> Type[Config]:
        return Config

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

    @command.new(name=lambda self: self.config["post_trigger"], require_subcommand=False)
    @command.argument("args", pass_raw=True, required=False)
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

        number = args_namespace.number
        hours = args_namespace.hours
        minutes = args_namespace.minutes
        days = args_namespace.days
        title = ' '.join(args_namespace.title) if args_namespace.title else None

        if (number and (hours or minutes or days)):
            await evt.reply("Please specify either a number of messages (-n) or a timeframe (-h/-m/-d), not both.")
            return

        if not (number or hours or minutes or days or evt.content.get_reply_to()):
            await evt.reply("Please reply to a message or specify either a number of messages (-n) or a timeframe (-h/-m/-d).")
            return

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
            replied_event = await evt.client.get_event(
                evt.room_id, evt.content.get_reply_to()
            )
            messages = [replied_event]

        if not messages:
            await evt.reply("No messages found to summarize.")
            return

        message_bodies = [event.content.body for event in reversed(messages) if hasattr(event.content, 'body')]
        combined_message = "\n\n".join(message_bodies)

        if messages:
            if len(messages) > 1:
                summary = await self.ai_integration.summarize_content(combined_message)
            else:
                head_summary = await self.ai_integration.summarize_content(message_bodies[0])
                if head_summary:
                    summary = f"{head_summary}\n\n---\n\nOriginal Message:\n{message_bodies[0]}"
                else:
                    summary = message_bodies[0]
        else:
            await evt.reply("Please reply to a message to summarize.")
            return

        if not summary:
            self.log.warning("AI summarization failed. Falling back to the original message content.")
            summary = combined_message

        ai_model_type = self.config["ai_model_type"]

        if ai_model_type == "none":
            if not title:
                await evt.reply(
                    "A title is required since AI model is set to 'none'. Please provide a title."
                )
                return
        else:
            if not title:
                title = await self.generate_title(summary)
            if not title:
                title = "Untitled Post"

        self.log.info(f"Generated Title: {title}")

        tags = await self.ai_integration.generate_tags(summary)
        if not tags:
            tags = ["bot-post"]

        if self.config["matrix_to_discourse_topic"]:
            category_id = self.config["matrix_to_discourse_topic"].get(evt.room_id)
        else:
            category_id = self.config["unsorted_category_id"]

        self.log.info(f"Using category ID: {category_id}")

        post_url, error = await self.discourse_api.create_post(
            title=title,
            raw=summary,
            category_id=category_id,
            tags=tags,
        )
        if post_url:
            posted_link_url = f"{self.config['discourse_base_url']}/tag/posted-link"
            post_url = post_url.replace("[", "").replace("]", "")
            await evt.reply(f"Forum post created with bypass links: {title}, {post_url} - See all community posted links {posted_link_url}")
        else:
            await evt.reply(f"Failed to create post: {error}")

    async def fetch_messages_by_number(self, evt: MessageEvent, number: int) -> List[MessageEvent]:
        messages = []
        prev_batch = None
        self.log.info(f"Using prev_batch token: {prev_batch}")

        try:
            sync_response = await self.client.sync(since=prev_batch)
            prev_batch = sync_response['next_batch']
        except Exception as e:
            self.log.error(f"Error during sync: {e}")
            return messages

        while len(messages) < number:
            try:
                response = await self.client.get_context(
                    room_id=evt.room_id,
                    from_token=prev_batch,
                    direction=PaginationDirection.BACKWARD,
                    limit=100
                )
            except Exception as e:
                self.log.error(f"Error fetching messages: {e}")
                break

            self.log.debug(f"Response: {response}")
            events = response.events
            if not events:
                break
            for event in events:
                if event.type == EventType.ROOM_MESSAGE and event.sender != self.client.mxid:
                    messages.append(event)
                    if len(messages) >= number:
                        break

            if hasattr(response, 'end'):
                prev_batch = response.end
            else:
                self.log.warning("No 'end' token in response, stopping pagination.")
                break

        return messages[:number]

    async def fetch_messages_by_time(self, evt: MessageEvent, time_delta: timedelta) -> List[MessageEvent]:
        messages = []
        prev_batch = None
        end_time = datetime.utcnow() - time_delta
        max_retries = 3
        retry_delay = 5

        try:
            sync_response = await self.client.sync(since=prev_batch)
            prev_batch = sync_response['next_batch']
        except Exception as e:
            self.log.error(f"Error during sync: {e}")
            return messages

        while True:
            for attempt in range(max_retries):
                try:
                    response = await self.client.get_context(
                        room_id=evt.room_id,
                        from_token=prev_batch,
                        direction=PaginationDirection.BACKWARD,
                        limit=100
                    )
                    break
                except Exception as e:
                    self.log.error(f"Error fetching messages: {e}")
                    if "504" in str(e):
                        self.log.warning(f"504 Gateway Timeout encountered. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        return messages
            else:
                self.log.error("Max retries reached. Exiting message fetch.")
                return messages

            self.log.debug(f"Response: {response}")
            events = response.events
            if not events:
                break

            for event in events:
                if event.type == EventType.ROOM_MESSAGE and event.sender != self.client.mxid:
                    event_time = datetime.fromtimestamp(event.server_timestamp / 1000, timezone.utc)
                    if event_time < end_time:
                        return messages
                    messages.append(event)

            if hasattr(response, 'end'):
                prev_batch = response.end
            else:
                self.log.warning("No 'end' token in response, stopping pagination.")
                break

        return messages

    async def generate_title_for_bypassed_links(self, message_body: str) -> Optional[str]:
        return await self.ai_integration.generate_links_title(message_body)

    async def generate_title(self, message_body: str) -> Optional[str]:
        return await self.ai_integration.generate_title(message_body, use_links_prompt=False)

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
                    for result in search_results:
                        result["views"] = result.get("views", 0)
                        result["created_at"] = result.get("created_at", "1970-01-01T00:00:00Z")

                    sorted_by_recent = sorted(
                        search_results, key=lambda x: x["created_at"], reverse=True
                    )
                    sorted_by_views = sorted(search_results, key=lambda x: x["views"], reverse=True)

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

    @event.on(EventType.ROOM_MESSAGE)
    async def handle_message(self, evt: MessageEvent) -> None:
        if evt.sender == self.client.mxid:
            return

        if evt.content.msgtype != MessageType.TEXT:
            return

        message_body = evt.content.body
        urls = extract_urls(message_body)
        if urls:
            await self.process_link(evt, message_body)

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

    def should_process_url(self, url: str) -> bool:
        for blacklist_pattern in self.config["url_blacklist"]:
            if re.match(blacklist_pattern, url, re.IGNORECASE):
                logger.debug(f"URL {url} matches blacklist pattern {blacklist_pattern}")
                return False

        for pattern in self.config["url_patterns"]:
            if re.match(pattern, url, re.IGNORECASE):
                logger.debug(f"URL {url} matches whitelist pattern {pattern}")
                return True

        return False

    async def process_link(self, evt: MessageEvent, message_body: str) -> None:
        urls = extract_urls(message_body)
        username = evt.sender.split(":")[0]

        urls_to_process = [url for url in urls if self.should_process_url(url)]

        if not urls_to_process:
            self.log.debug("No URLs matched the configured patterns or all URLs were blacklisted")
            await evt.reply("No valid URLs found to process.")
            return

        for url in urls_to_process:
            duplicate_exists = await self.discourse_api.check_for_duplicate(url)
            if duplicate_exists:
                await evt.reply(f"A post with this URL already exists: {url}")
                continue

            content = await scrape_content(url)
            if content:
                summary = await self.ai_integration.summarize_content(content, user_message=message_body)
                if not summary:
                    self.log.warning(f"Summarization failed for URL: {url}")
                    summary = "Content could not be scraped or summarized."
            else:
                self.log.warning(f"Scraping content failed for URL: {url}")
                summary = "Content could not be scraped or summarized."

            title = await self.ai_integration.generate_links_title(message_body)
            if not title:
                self.log.info(f"Generating title using URL and domain for: {url}")
                title = await self.ai_integration.generate_links_title(f"URL: {url}, Domain: {url.split('/')[2]}")
                if not title:
                    title = "Untitled Post"

            tags = await self.ai_integration.generate_tags(content)
            if "posted-link" not in tags:
                tags.append("posted-link")
            if not tags:
                tags = ["posted-link"]

            bypass_links = generate_bypass_links(url)

            post_body = (
                f"{summary}\n\n"
                f"**Original Link:** <{url}>\n\n"
                f"**12ft.io Link:** [12ft.io]({bypass_links['12ft']})\n"
                f"**Archive.org Link:** [Archive.org]({bypass_links['archive']})\n\n"
                f"User Message: {message_body}\n\n"
                f"for more on see the [post on bypassing methods](https://forum.irregularchat.com/t/bypass-links-and-methods/98?u=sac)"
            )

            category_id = self.config["matrix_to_discourse_topic"].get(evt.room_id, self.config["unsorted_category_id"])

            self.log.info(f"Using category ID: {category_id}")

            post_url, error = await self.discourse_api.create_post(
                title=title,
                raw=post_body,
                category_id=category_id,
                tags=tags,
            )

            if post_url:
                posted_link_url = f"{self.config['discourse_base_url']}/tag/posted-link"
                post_url = post_url.replace("[", "").replace("]", "")
                await evt.reply(f"🔗Forum post created with bypass links: {title}, {post_url}")
            else:
                await evt.reply(f"Failed to create post: {error}")