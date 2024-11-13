import asyncio
import json
import re
from datetime import datetime
from typing import Type, Dict
from mautrix.client import Client
from mautrix.types import Format, TextMessageEventContent, EventType, RelationType
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event
import aiohttp
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

# Config class to manage configuration
class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        # General configuration
        ## AI Model Configuration. importing all the AI model configuration from the base-config.yaml file to check which AI model is selected
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


class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.config.load_and_update()
        self.log.info("MatrixToDiscourseBot started")

    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    @command.new(name="fpost", require_subcommand=False)
    @command.argument("title", pass_raw=True, required=False)  # Title is optional and taken from AI if not provided
    async def post_to_discourse(self, evt: MessageEvent, title: str = None) -> None:
        self.log.info("Command !fpost triggered.")
        await evt.reply("Creating a Forum post, log in to the community discourse to view all posts and to engage on the forum...")

        try:
            self.log.info(f"Received event body: {evt.content.body}")

            # Check if the message is a reply to another message
            if not evt.content.get_reply_to():
                await evt.reply("You must reply to a message to use this command.")
                return

            # Extract the body of the replied-to message
            replied_event = await evt.client.get_event(evt.room_id, evt.content.get_reply_to())
            message_body = replied_event.content.body
            self.log.info(f"Message body: {message_body}")

            # Determine if a title is required
            ai_model_type = self.config["ai_model_type"]

            if ai_model_type == "none":
                # If AI is set to 'none', title is required from the user
                if not title:
                    await evt.reply("A title is required since AI model is set to 'none'. Please provide a title.")
                    return
            else:
                # Use provided title or generate one using the AI model
                if not title:
                    title = await self.generate_title(message_body)
                if not title:
                    title = "Default Title"  # Fallback title if generation fails

            self.log.info(f"Generated Title: {title}")

            # Get the topic ID based on the room ID
            topic_id = self.config["matrix_to_discourse_topic"].get(evt.room_id, self.config["unsorted_category_id"])

            # Create the post on Discourse
            post_url, error = await self.create_post(
                self.config["discourse_base_url"], 
                topic_id, 
                title, 
                message_body
            )
            if post_url:
                await evt.reply(f"Post created successfully! URL: {post_url} \n\n Log in to the community discourse to engage on the forum.")
            else:
                await evt.reply(f"Failed to create post: {error}")

        except Exception as e:
            self.log.error(f"Error processing !fpost command: {e}")
            await evt.reply(f"An error occurred: {e}")

    async def generate_title(self, message_body: str) -> str:
        ai_model_type = self.config["ai_model_type"]
        
        if ai_model_type == "openai":
            return await self.generate_openai_title(message_body)
        elif ai_model_type == "local":
            return await self.generate_local_title(message_body)
        elif ai_model_type == "google":
            return await self.generate_google_title(message_body)
        return None  # If none is selected, this should never be called

    async def generate_openai_title(self, message_body: str) -> str:
        prompt = f"Create a brief (3-10 word) attention grabbing title for the following post on the community forum: {message_body}"
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['openai.api_key']}"
            }
            data = {
                "model": self.config["openai.model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config["openai.max_tokens"],
                "temperature": self.config["openai.temperature"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["openai.api_endpoint"], headers=headers, json=data) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        self.log.error(f"OpenAI API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.log.error(f"Error generating title with OpenAI: {e}")
            return None

    async def generate_local_title(self, message_body: str) -> str:
        # Simulate a local LLM call or use an external local API
        prompt = f"Generate a title for: {message_body}"
        try:
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": self.config["local_llm.model_path"],
                "prompt": prompt
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["local_llm.api_endpoint"], headers=headers, json=data) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        return response_json["title"].strip()
                    else:
                        self.log.error(f"Local LLM API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.log.error(f"Error generating title with Local LLM: {e}")
            return None

    async def generate_google_title(self, message_body: str) -> str:
        prompt = f"Create a brief (3-10 word) attention grabbing title for the following post on the community forum: {message_body}"
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['google.api_key']}"
            }
            data = {
                "model": self.config["google.model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config["google.max_tokens"],
                "temperature": self.config["google.temperature"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["google.api_endpoint"], headers=headers, json=data) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        self.log.error(f"Google API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.log.error(f"Error generating title with Google Gemini: {e}")
            return None

    async def create_post(self, base_url, category_id, title, message_body):
        url = f"{base_url}/posts.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"]
        }
        payload = {
            "title": title,
            "raw": message_body,
            "category": category_id
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_text = await response.text()
                if response.status == 200:
                    data = await response.json()
                    topic_id = data.get("topic_id")
                    topic_slug = data.get("topic_slug")
                    post_url = f"{base_url}/t/{topic_slug}/{topic_id}" if topic_id and topic_slug else "URL not available"
                    return post_url, None
                else:
                    return None, f"Failed to create post: {response.status}\nResponse: {response_text}"

    @command.new(name="fsearch", require_subcommand=False)
    @command.argument("query", pass_raw=True, required=True)
    async def search_discourse(self, evt: MessageEvent, query: str) -> None:
        self.log.info("Command !fsearch triggered.")
        await evt.reply("Searching the forum...")

        try:
            search_url = f"{self.config['discourse_base_url']}/search.json"
            headers = {
                "Content-Type": "application/json",
                "Api-Key": self.config["discourse_api_key"],
                "Api-Username": self.config["discourse_api_username"]
            }
            params = {"q": query}

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        search_results = response_json.get("topics", [])

                        # Safely get keys with default values
                        for result in search_results:
                            result['views'] = result.get('views', 0)
                            result['created_at'] = result.get('created_at', '1970-01-01T00:00:00Z')

                        # Sort search results by created_at for most recent and by views for most seen
                        sorted_by_recent = sorted(search_results, key=lambda x: x['created_at'], reverse=True)
                        sorted_by_views = sorted(search_results, key=lambda x: x['views'], reverse=True)

                        # Select top 2 most recent and top 3 most seen
                        top_recent = sorted_by_recent[:2]
                        top_seen = sorted_by_views[:3]

                        def format_results(results):
                            return "\n".join([f"* [{result['title']}]({self.config['discourse_base_url']}/t/{result['slug']}/{result['id']})" for result in results])

                        result_msg = (
                            "**Top 2 Most Recent:**\n" +
                            format_results(top_recent) +
                            "\n\n**Top 3 Most Seen:**\n" +
                            format_results(top_seen)
                        )

                        if search_results:
                            await evt.reply(f"Search results:\n{result_msg}")
                        else:
                            await evt.reply("No results found.")
                    else:
                        self.log.error(f"Discourse API error: {response.status} {response_json}")
                        await evt.reply("Failed to perform search.")
        except Exception as e:
            self.log.error(f"Error processing !fsearch command: {e}")
            await evt.reply(f"An error occurred: {e}")