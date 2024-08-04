import asyncio
import aiohttp
import json
import yaml

# Load configuration from YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

async def create_post(config):
    url = f"{config['discourse_base_url']}/posts.json"
    api_key = config["discourse_api_key"]
    api_username = config["discourse_api_username"]

    headers = {
        "Content-Type": "application/json",
        "Api-Key": api_key,
        "Api-Username": api_username
    }
    payload = {
        "title": "Test Title",
        "raw": "Test message body",
        "category": config["unsorted_category_id"],  # Use the unsorted category ID from config
        "tags": ["test"]
    }

    # Print out the variables to debug
    print("URL:", url)
    print("Headers:", json.dumps(headers, indent=2))
    print("Payload:", json.dumps(payload, indent=2))

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response_text = await response.text()
            if response.status == 200:
                data = await response.json()
                print("Post created successfully:", data)
                topic_id = data.get("topic_id")
                topic_slug = data.get("topic_slug")
                post_url = f"{config['discourse_base_url']}/t/{topic_slug}/{topic_id}" if topic_id and topic_slug else "URL not available"
                print("Post URL:", post_url)
            else:
                print("Failed to create post", response.status)
                print("Response:", response_text)

if __name__ == "__main__":
    # Load configuration
    config = load_config('base-config.yml')
    print("Loaded Configuration:", json.dumps(config, indent=2))

    # Run the create_post function with the loaded configuration
    asyncio.run(create_post(config))
