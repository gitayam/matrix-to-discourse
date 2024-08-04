# Matrix to Discourse Maubot Plugin

This Maubot plugin allows you to post messages from Matrix chat rooms to a Discourse forum. When a user sends a command in a Matrix room, the bot will create a corresponding post in the specified Discourse topic.

## Features

- Listen for commands in Matrix chat rooms.
- Create posts in specified Discourse topics.
- Return the URL of the newly created Discourse post.

## Requirements

- A Maubot instance set up and running.
- Discourse API key and username with permissions to create posts.
- A mapping of Matrix chat room IDs to Discourse topic IDs.

## Installation

### Step 1: Setup the Environment

1. Clone the repository or copy the plugin files to a suitable location.
2. Ensure you have the necessary dependencies listed in `requirements.txt`.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Build the Plugin

Use `mbc build` to build the plugin:

```bash
mbc build
```

Alternatively, you can manually zip the files:

```bash
zip -r matrix_to_discourse.mbp *
```

### Step 4: Upload the Plugin

Upload the `.mbp` file to your Maubot instance via the Maubot web interface.

### Step 5: Configure the Plugin

Set the following configuration values in the Maubot interface:

- `discourse_api_key`: Your Discourse API key.
- `discourse_api_username`: Your Discourse API username.
- `discourse_base_url`: The base URL of your Discourse instance.
- `signal_to_discourse_topic`: A mapping of Matrix chat room IDs to Discourse topic IDs. This should be in JSON format.

Example configuration:

```json
{
    "discourse_api_key": "your_discourse_api_key",
    "discourse_api_username": "your_discourse_api_username",
    "discourse_base_url": "https://your.discourse.instance",
    "signal_to_discourse_topic": {
        "!roomID1:server": "topic_id_1",
        "!roomID2:server": "topic_id_2"
    }
}
```

## Usage

In any Matrix room that the bot is a member of, use the following command to create a post in the corresponding Discourse topic:

```plaintext
!post Your message here
```

The bot will create a post with the specified message and reply with the URL of the newly created post.

## File Structure

```
matrix_to_discourse/
├── maubot.yaml
├── requirements.txt
└── matrix_to_discourse/
    ├── __init__.py
    └── discourse_post.py
```

### `maubot.yaml`

```yaml
maubot: 0.1.0
id: com.example.matrix_to_discourse
version: 1.0.0
license: MIT
modules:
  - matrix_to_discourse
main_class: MatrixToDiscourseBot
dependencies:
  - aiohttp
  - python-dotenv
config:
  required: true
  fields:
    discourse_api_key:
      type: str
      description: The API key for the Discourse instance.
    discourse_api_username:
      type: str
      description: The API username for the Discourse instance.
    discourse_base_url:
      type: str
      description: The base URL for the Discourse instance.
    signal_to_discourse_topic:
      type: dict
      description: A mapping of Signal chat identifiers to Discourse topic IDs.
```

### `requirements.txt`

```
aiohttp
python-dotenv
```

### `matrix_to_discourse/__init__.py`

```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command
from .discourse_post import create_post

class MatrixToDiscourseBot(Plugin):
    @command.new(name="post", help="Post a message to Discourse")
    async def post_to_discourse(self, evt: MessageEvent) -> None:
        chat_identifier = evt.room_id
        title = evt.content.body
        
        # Create the post
        post_url = await create_post(self.config, chat_identifier, title)
        
        if post_url:
            await evt.reply(f"Post created successfully! URL: {post_url}")
        else:
            await evt.reply("Failed to create post.")
```

### `matrix_to_discourse/discourse_post.py`

```python
import aiohttp

async def create_post(config, chat_identifier, title):
    topic_id = config['signal_to_discourse_topic'].get(chat_identifier)
    if not topic_id:
        return None

    url = f"{config['discourse_base_url']}/posts.json"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": config['discourse_api_key'],
        "Api-Username": config['discourse_api_username']
    }
    payload = {
        "topic_id": topic_id,
        "raw": title
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                post_data = await response.json()
                post_url = f"{config['discourse_base_url']}/t/{post_data['topic_slug']}/{post_data['topic_id']}/{post_data['post_number']}"
                return post_url
            else:
                return None
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any changes.

## License

This project is licensed under the MIT License.
