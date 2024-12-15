# ai_integration.py
import aiohttp
import json
import traceback
from typing import Optional, List

from plugin.discourse_api import DiscourseAPI

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

    async def generate_links_title(self, message_body: str) -> Optional[str]:
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

    TAG_PROMPT = """Analyze the following content and suggest 2-5 relevant tags {user_message}.  
        NOW Choose from or be inspired by these existing tags: {tag_list}
        If none of the existing tags fit well, suggest new related tags.
        The tags should be lowercase, use hyphens instead of spaces, and be concise.       

        Return only the tags as a comma-separated list, no explanation needed."""

    async def generate_tags(self, user_message: str = "") -> Optional[List[str]]:
        try:
            if not self.discourse_api:
                self.log.error("Discourse API is not initialized.")
                return None

            # Get existing tags from Discourse for context
            all_tags = await self.discourse_api.get_all_tags()
            if all_tags is None:
                self.log.error("Failed to fetch tags from Discourse API.")
                return ["posted-link"]  # Fallback to a default tag

            self.log.debug(f"Type of all_tags: {type(all_tags)}")
            self.log.debug(f"Content of all_tags: {all_tags}")

            if not isinstance(all_tags, list) or not all(isinstance(tag, dict) for tag in all_tags):
                self.log.error("Unexpected format for all_tags. Expected a list of dictionaries.")
                return ["posted-link"]

            tag_names = [tag["name"] for tag in all_tags]
            tag_list = ", ".join(list(dict.fromkeys(tag_names)))

            prompt = self.TAG_PROMPT.format(tag_list=tag_list, user_message=user_message)

            if self.config["ai_model_type"] == "openai":
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
                                tags_text = response_json["choices"][0]["message"]["content"].strip()
                                tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
                                tags = [tag for tag in tags if tag][:5]
                                return tags
                            else:
                                self.log.error(f"OpenAI API error: {response.status} {response_json}")
                                return None
                except Exception as e:
                    tb = traceback.format_exc()
                    self.log.error(f"Error generating tags with OpenAI: {e}\n{tb}")
                    return None

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
                            return None

                        if response.status == 200:
                            tags_text = response_json.get("text", "").strip()
                            tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
                            tags = [tag for tag in tags if tag][:5]
                            return tags
                        else:
                            self.log.error(f"Local LLM API error: {response.status} {response_json}")
                            return None

            elif self.config["ai_model_type"] == "google":
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
                            return None

                        if response.status == 200:
                            tags_text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                            tags = [tag.strip().lower().replace(' ', '-') for tag in tags_text.split(',')]
                            tags = [tag for tag in tags if tag][:5]
                            return tags
                        else:
                            self.log.error(f"Google API error: {response.status} {response_json}")
                            return None

            else:
                self.log.error(f"Unknown AI model type: {self.config['ai_model_type']}")
                return None

        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error generating tags: {e}\n{tb}")
            return ["fix-me"]

    generate_title_prompt = "Create a brief (3-10 word) attention-grabbing title for the {target_audience} for the following post on the community forum: {message_body}"
    generate_links_title_prompt = "Create a brief (3-10 word) attention-grabbing title for the following post on the community forum include the source and title of the linked content: {message_body}"

    async def generate_openai_title(self, message_body: str, use_links_prompt: bool = False) -> Optional[str]:
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
        if use_links_prompt:
            prompt = self.generate_links_title_prompt.format(message_body=message_body)
        else:
            prompt = self.generate_title_prompt.format(target_audience=self.target_audience, message_body=message_body)
        try:
            # Implement local LLM API if needed
            pass
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error generating title with local LLM: {e}\n{tb}")
            return None

    async def summarize_with_local_llm(self, content: str) -> Optional[str]:
        prompt = f"Please provide a concise summary which is relevant to the {self.target_audience} of the following content:\n\n{content}"
        try:
            # Implement local LLM API if needed
            pass
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
                        return response_json["candidates"][0]["output"].strip()
                    else:
                        self.log.error(f"Google API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error summarizing with Google: {e}\n{tb}")
            return None