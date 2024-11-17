# ai_integration.py

import aiohttp
import traceback


class AIIntegration:
    def __init__(self, config, log):
        self.config = config
        self.log = log

    async def generate_title(self, message_body: str) -> str:
        ai_model_type = self.config["ai_model_type"]

        if ai_model_type == "openai":
            return await self.generate_openai_title(message_body)
        elif ai_model_type == "local":
            return await self.generate_local_title(message_body)
        elif ai_model_type == "google":
            return await self.generate_google_title(message_body)
        return None  # If none is selected, this should never be called

    async def summarize_content(self, content: str) -> str:
        ai_model_type = self.config["ai_model_type"]
        if ai_model_type == "openai":
            return await self.summarize_with_openai(content)
        elif ai_model_type == "local":
            return await self.summarize_with_local_llm(content)
        elif ai_model_type == "google":
            return await self.summarize_with_google(content)
        return None

    # Implement the methods for each AI model
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
            tb = traceback.format_exc()
            self.log.error(f"Error generating title with OpenAI: {e}\n{tb}")
            return None

    async def summarize_with_openai(self, content: str) -> str:
        prompt = f"Please provide a concise summary of the following content:\n\n{content}"
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
            tb = traceback.format_exc()
            self.log.error(f"Error summarizing with OpenAI: {e}\n{tb}")
            return None

    async def generate_local_title(self, message_body: str) -> str:
        # Implement according to local LLM API using openai url but hosted locally or in private ip
        # This is a placeholder

        return None

    async def summarize_with_local_llm(self, content: str) -> str:
        # Implement according to local LLM API
        # This is a placeholder
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
            tb = traceback.format_exc()
            self.log.error(f"Error generating title with Google: {e}\n{tb}")
            return None

    async def summarize_with_google(self, content: str) -> str:
        prompt = f"Please provide a concise summary of the following content:\n\n{content}"
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
            tb = traceback.format_exc()
            self.log.error(f"Error summarizing with Google: {e}\n{tb}")
            return None