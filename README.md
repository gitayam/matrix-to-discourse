# Matrix to Discourse Bot

A Maubot plugin that allows users to post messages from a Matrix room to a Discourse forum and search the forum.


## Typical Use
To use this bot you'll need to have:
- Maubot installed and running on your Matrix server
  - For the most robust and easiest matrix and maubot setup, consider using: https://github.com/spantaleev/matrix-docker-ansible-deploy
- A Discourse forum that you can post to
- An OpenAI API key

Download the plugin.mbp file from the [releases page](https://github.com/gitayam/matrix-to-discourse/releases/latest) and follow the instructions in the Maubot documentation to install the plugin.

You'll upload the plugin to your Maubot instance, configure it with your OpenAI API key, Discourse API key, and Discourse categories and information and then you can use the bot to post messages from Matrix to Discourse and search Discourse from Matrix.

Maubot Instance is typically at : https://matrix.domain.tld/_matrix/maubot/
If you are using the matrix-docker-ansible-deploy, set up using:
```yml
matrix_bot_maubot_enabled: true
matrix_bot_maubot_initial_password: 'secrets_here'
matrix_bot_maubot_admins:
  - user: 'secrets_here'
```

## Configuration

### OpenAI API Configuration

To get your API key, sign up at [OpenAI Platform](https://platform.openai.com/signup) or log in at [OpenAI Login](https://platform.openai.com/login).

- `gpt_api_key`: Your OpenAI API key.
- `api_endpoint`: API endpoint for OpenAI.
- `model`: Model to use for OpenAI.
- `max_tokens`: Maximum tokens for OpenAI responses.
- `temperature`: Temperature setting for OpenAI responses.

### Discourse Configuration

To get your API key, go to `https://discourse.example.com/admin/api/keys`. Replace `discourse.example.com` with your Discourse URL.

- `discourse_api_key`: Your Discourse API key.
- `discourse_api_username`: Your Discourse API username.
- `discourse_base_url`: Base URL for your Discourse forum.
- `matrix_to_discourse_topic`: Mapping of Matrix room IDs to Discourse topic IDs.
- `unsorted_category_id`: The category ID for the unsorted category.

### Example `base-config.yaml`

```yaml
###### OpenAI API Configuration ######
gpt_api_key: "gpt_api_key"
api_endpoint: "https://api.openai.com/v1/chat/completions"
model: "gpt-4o-mini"
max_tokens: 3000
temperature: 1

###### Discourse Configuration ######
discourse_api_key: "discourse_api_key"
discourse_api_username: "discourse_api_username"
discourse_base_url: "https://discourse.example.com"
matrix_to_discourse_topic:
  "!roomID1:server": "27"
  "!roomID2:server": "4"
unsorted_category_id: 9
```

## Commands

### `!fpost`

Creates a post on the Discourse forum using the replied-to message's content.
- **Usage:** `!fpost` 
This will prompt GPT to generate a title based on the body. 
#TODO gracefully handle missing gpt api

- **Usage:** `!fpost [title]`
- **Example:**
  - Reply to a message with `!fpost My Custom Title`

### `!fsearch`

Searches the Discourse forum for the specified query.

- **Usage:** `!fsearch <query>`
- **Example:**
  - `!fsearch matrix bots`

## Installation

1. Clone the repository.
2. Install the required dependencies.
3. Deploy the bot on your Maubot instance.
4. Configure the `base-config.yaml` file with your settings.

## Contributing
To contribute to this project, please see the [Roadmap](./ROADMAP.md) page for a list of features that need to be implemented.
## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](./LICENSE) file for details.
