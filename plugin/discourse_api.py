# discourse_api.py
import json
import re
import aiohttp
import traceback

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

    async def check_for_duplicate(self, url: str, tag: str = "posted-link") -> bool:
        search_url = f"{self.config['discourse_base_url']}/search/query.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.config["discourse_api_key"],
            "Api-Username": self.config["discourse_api_username"],
        }

        base_url = re.sub(r'\?.*$', '', url)
        self.log.debug(f"Checking for duplicates of base URL: {base_url}")

        params = {"term": f'tags:{tag}'}
        self.log.debug(f"Searching Discourse with params: {params}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        self.log.error(f"Discourse API error: {response.status}")
                        return False

                    response_json = await response.json()
                    self.log.debug(f"Received response: {response_json}")

                    for post in response_json.get("posts", []):
                        raw_content = post.get("raw", "")
                        self.log.debug(f"Checking post ID {post.get('id')} with raw content: {raw_content}")

                        if "Original Link:" in raw_content and base_url in raw_content:
                            self.log.info(f"Duplicate post found: {post.get('topic_id')}")
                            return True

            except Exception as e:
                self.log.error(f"Error during Discourse API request: {e}")
                return False

        self.log.info("No duplicate found.")
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

    async def get_top_tags(self):
        num_tags = 10
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
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        return response_json[:num_tags]
                    else:
                        self.log.error(f"Discourse API error: {response.status} {response_json}")
                        return None
        except Exception as e:
            self.log.error(f"Error fetching top tags: {e}")
            return None

    async def get_all_tags(self):
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
                    self.log.debug(f"Response text: {response_text}")
                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.log.error(f"Error decoding Discourse response: {e}\nResponse text: {response_text}")
                        return None

                    if response.status == 200:
                        self.log.debug(f"Response JSON: {response_json}")
                        if isinstance(response_json, dict) and "tags" in response_json:
                            tags = response_json["tags"]
                            if isinstance(tags, list):
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