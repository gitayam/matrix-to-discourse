
# Matrix to Discourse Bot 
[v1.1](https://github.com/gitayam/matrix-to-discourse/releases/tag/v1.1)

A Maubot plugin that allows users to post messages from a Matrix room to a Discourse forum and search the forum. This version includes support for multiple AI API options, such as OpenAI, Google Gemini, local LLM models, and the ability to disable AI altogether.

## Typical Use

To use this bot you’ll need:

	•	Maubot installed and running on your Matrix server
	•	For the most robust and easiest matrix and maubot setup, consider using: matrix-docker-ansible-deploy
	•	A Discourse forum that you can post to
	•	An AI API key (OpenAI, Google Gemini, or a local LLM model, or you can configure it to use none)

Download the matrix-to-discourse.mbp file from the releases page and follow the instructions in the Maubot documentation to install the plugin.

You’ll upload the plugin to your Maubot instance, configure it with your AI API key (if applicable), Discourse API key, and Discourse categories and information. You can then use the bot to post messages from Matrix to Discourse and search Discourse from Matrix.

Maubot Instance is typically at: https://matrix.domain.tld/_matrix/maubot/

If using matrix-docker-ansible-deploy, set it up with the following configuration:

```
matrix_bot_maubot_enabled: true
matrix_bot_maubot_initial_password: 'secrets_here'
matrix_bot_maubot_admins:
  - user: 'secrets_here'
```

## AI Model Options

You can configure the bot to use one of the following AI models for generating titles for Discourse posts:

	•	OpenAI: Uses OpenAI models like GPT.
	•	Google Gemini: Uses Google Gemini for AI-powered tasks.
	•	Local LLM: Uses a locally hosted LLM model or a local server such as Ollama.
	•	None: If no AI model is configured, the user must provide a title for Discourse posts manually.

### AI Model Configuration

	•	ai_model_type: Choose the AI model type. Options are openai, google, local, or none.
	•	If none, the user must provide a title for each post.

#### OpenAI API Configuration

To get your API key, sign up at OpenAI Platform or log in at OpenAI Login.

	•	gpt_api_key: Your OpenAI API key.
	•	api_endpoint: API endpoint for OpenAI.
	•	model: Model to use for OpenAI (e.g., gpt-4).
	•	max_tokens: Maximum tokens for OpenAI responses.
	•	temperature: Temperature setting for OpenAI responses.

#### Google Gemini API Configuration

To use the Google Gemini API, follow the steps to get an API key from Google Cloud Console.

	•	google_api_key: Your Google Gemini API key.
	•	api_endpoint: API endpoint for Google Gemini.
	•	model: Google Gemini model (e.g., gemini-1.5-pro).
	•	max_tokens: Maximum tokens for responses.
	•	temperature: Temperature setting for responses.

#### Local LLM Configuration

To use a local model (e.g., LLaMA or a model hosted on Ollama), provide the local API details or file path.

	•	model_path: Path to the local LLM model (e.g., .gguf files).
	•	api_endpoint: Endpoint for your local LLM server (e.g., http://localhost:11434/api/chat).
	•	model: Local LLM model name.

Discourse Configuration

To get your API key, go to https://discourse.example.com/admin/api/keys. Replace discourse.example.com with your Discourse URL.

	•	discourse_api_key: Your Discourse API key.
	•	discourse_api_username: Your Discourse API username.
	•	discourse_base_url: Base URL for your Discourse forum.
	•	matrix_to_discourse_topic: Mapping of Matrix room IDs to Discourse topic IDs.
	•	unsorted_category_id: The category ID for the unsorted category.

Example base-config.yaml

###### AI Model Configuration ######
ai_model_type: "openai"  # Choose between "openai", "google", "local", "none"

###### OpenAI API Configuration ######
```
gpt_api_key: "your_openai_key_here"
api_endpoint: "https://api.openai.com/v1/chat/completions"
model: "gpt-4"
max_tokens: 3000
temperature: 1
```

###### Google Gemini API Configuration ######
``` 
google_api_key: "your_google_gemini_key_here"
api_endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
model: "gemini-1.5-pro"
max_tokens: 3000
temperature: 1
```

###### Local LLM Configuration ######
```
model_path: "/path/to/your/local/model"
api_endpoint: "http://localhost:11434/api/chat"
model: "llama2"
max_tokens: 3000
temperature: 1
```

###### Discourse Configuration ######
```
discourse_api_key: "your_discourse_key_here"
discourse_api_username: "your_discourse_username_here"
discourse_base_url: "https://discourse.example.com"
matrix_to_discourse_topic:
  "!roomID1:server": "27"
  "!roomID2:server": "4"
unsorted_category_id: 9
```

Commands

`!fpost`

Creates a post on the Discourse forum using the replied-to message’s content.

	•	Usage:
	•	!fpost [title] — Posts the replied-to message with the given title to Discourse.
	•	If AI is enabled, the bot will generate a title if one is not provided.
	•	If AI is set to none, the user must provide a title manually.

!fsearch

Searches the Discourse forum for the specified query.

	•	Usage: !fsearch <query>
	•	Example:
	•	!fsearch matrix bots

Installation

	1.	Clone the repository:

git clone https://github.com/gitayam/matrix-to-discourse.git


	2.	Install the required dependencies:

pip3 install -r requirements.txt


	3.	Deploy the bot on your Maubot instance.
	4.	Configure the base-config.yaml file with your settings.

Contributing

To contribute to this project, please see the Roadmap page for a list of features that need to be implemented.

License

This project is licensed under the GPL-3.0 License. See the LICENSE file for details.

