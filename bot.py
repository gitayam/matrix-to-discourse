import aiohttp
from maubot import Plugin, MessageEvent
from maubot.handlers import command

class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.log.info("MatrixToDiscourseBot started")

    @command.new(name="post")
    async def post_to_discourse(self, evt: MessageEvent) -> None:
        self.log.info("Command !post triggered.")
        await evt.reply("Thanks")

        try:
            self.log.info(f"Received event body: {evt.content.body}")
            await evt.reply(f"Debugging: Received event body: {evt.content.body}")

            # Check if the message is a reply to another message
            if not evt.content.get_reply_to():
                await evt.reply("Debugging: You must reply to a message to use this command.")
                return

            # Extract the body of the replied-to message
            replied_event = await evt.client.get_event(evt.room_id, evt.content.get_reply_to())
            message_body = replied_event.content.body
            self.log.info(f"Message body: {message_body}")
            await evt.reply(f"Debugging: Message body: {message_body}")

            # Extract title before the !post command
            command_parts = evt.content.body.strip().rsplit('!post', 1)
            if len(command_parts) > 1:
                title = command_parts[0].strip()
            else:
                title = "testing post"
            self.log.info(f"Title: {title}")
            await evt.reply(f"Debugging: Title: {title}")

            # Hardcoded configuration values
            config = {
                "discourse_api_key": "a8a21514dc11edab80e31276417737c85778ca6c971cedba5400a3662b5709fd",
                "discourse_api_username": "sac",
                "discourse_base_url": "https://forum.irregularchat.com",
                "unsorted_category_id": 9
            }
            self.log.info(f"Config: {config}")
            await evt.reply(f"Debugging: Config: {config}")

            post_url, error = await self.create_post(config, config['unsorted_category_id'], title, message_body)
            if post_url:
                await evt.reply(f"Post created successfully! URL: {post_url}")
            else:
                await evt.reply(f"Failed to create post: {error}")

        except Exception as e:
            self.log.error(f"Error processing !post command: {e}")
            await evt.reply(f"Debugging: An error occurred: {e}")

    async def create_post(self, config, category_id, title, message_body):
        url = f"{config['discourse_base_url']}/posts.json"
        headers = {
            "Content-Type": "application/json",
            "Api-Key": config["discourse_api_key"],
            "Api-Username": config["discourse_api_username"]
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
                    post_url = f"{config['discourse_base_url']}/t/{topic_slug}/{topic_id}" if topic_id and topic_slug else "URL not available"
                    return post_url, None
                else:
                    return None, f"Failed to create post: {response.status}\nResponse: {response_text}"
