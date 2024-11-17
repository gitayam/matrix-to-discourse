# tests/test_bot.py

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from bot import MatrixToDiscourseBot
from maubot import MessageEvent
import asyncio

# Import missing modules
from url_handler import extract_urls, generate_bypass_links
from discourse_api import DiscourseAPI


class TestMatrixToDiscourseBot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mocking the required dependencies for the Plugin base class
        self.mock_client = AsyncMock()
        self.mock_loop = asyncio.get_event_loop()
        self.mock_http = MagicMock()
        self.mock_instance_id = "test_instance"
        self.mock_log = MagicMock()
        self.mock_config = MagicMock()
        self.mock_database = MagicMock()
        self.mock_webapp = MagicMock()
        self.mock_webapp_url = "http://localhost"
        self.mock_loader = MagicMock()

        # Mock the configuration
        self.mock_config.__getitem__.side_effect = lambda key: {
            "ai_model_type": "openai",
            "openai.api_key": "test_openai_key",
            "openai.api_endpoint": "https://api.openai.com/v1/chat/completions",
            "openai.model": "gpt-4",
            "openai.max_tokens": 3000,
            "openai.temperature": 1,
            "discourse_api_key": "test_discourse_api_key",
            "discourse_api_username": "test_user",
            "discourse_base_url": "https://discourse.example.com",
            "unsorted_category_id": "1",
            "matrix_to_discourse_topic": {},
            "search_trigger": "fsearch",
            "post_trigger": "fpost",
            "help_trigger": "fhelp",
            "url_post_trigger": "furl",
            "url_patterns": [r'https?://example\.com/.*'],
        }[key]

        # Initialize the bot with mocked dependencies
        self.bot = MatrixToDiscourseBot(
            client=self.mock_client,
            loop=self.mock_loop,
            http=self.mock_http,
            instance_id=self.mock_instance_id,
            log=self.mock_log,
            config=self.mock_config,
            database=self.mock_database,
            webapp=self.mock_webapp,
            webapp_url=self.mock_webapp_url,
            loader=self.mock_loader,
        )

        # Mock additional attributes
        self.bot.discourse_api = AsyncMock()
        self.bot.ai_integration = AsyncMock()

    async def test_help_command(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        evt.content = MagicMock()
        evt.content.body = "!fhelp"

        # Call the implementation directly
        await self.bot.handle_help(evt)

        # Assert the reply is sent
        evt.reply.assert_called_once_with(
            "Welcome to the Community Forum Bot!\n\n"
            "To create a post on the forum, reply to a message with `!fpost`.\n"
            "To search the forum, use `!fsearch <query>`.\n"
            "To post a URL, reply to a message containing a URL with `!furl`.\n"
            "For help, use `!fhelp`."
        )

    async def test_post_to_discourse_with_title(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        evt.content = MagicMock()
        evt.content.get_reply_to.return_value = "event_id"
        evt.content.body = "!fpost Sample Title"
        evt.client = AsyncMock()
        evt.client.get_event.return_value = MagicMock()
        evt.client.get_event.return_value.content.body = "This is a sample body"
        evt.room_id = "!roomid:example.com"

        # Mock Discourse API response
        self.bot.discourse_api.create_post.return_value = ("http://example.com", None)

        # Call the implementation directly
        await self.bot.handle_post_to_discourse(evt, title='Sample Title')

        # Assert replies
        evt.reply.assert_any_call(
            "Creating a Forum post, log in to the community forum to view all posts and to engage on the forum..."
        )
        evt.reply.assert_any_call(
            "Post created successfully! URL: http://example.com \n\n Log in to the community to engage with this post."
        )

    async def test_post_to_discourse_without_title(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        evt.content = MagicMock()
        evt.content.get_reply_to.return_value = "event_id"
        evt.content.body = "!fpost"
        evt.client = AsyncMock()
        evt.client.get_event.return_value = MagicMock()
        evt.client.get_event.return_value.content.body = "This is the message body"
        evt.room_id = "!roomid:example.com"

        # Mock AI integration and Discourse API
        self.bot.ai_integration.generate_title.return_value = "Generated Title"
        self.bot.discourse_api.create_post.return_value = ("https://discourse.example.com/t/test-topic/1", None)

        # Call the implementation directly
        await self.bot.handle_post_to_discourse(evt)

        # Assert replies
        evt.reply.assert_any_call(
            "Creating a Forum post, log in to the community forum to view all posts and to engage on the forum..."
        )
        evt.reply.assert_any_call(
            "Post created successfully! URL: https://discourse.example.com/t/test-topic/1 \n\n Log in to the community to engage with this post."
        )

    async def test_search_discourse_with_results(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        evt.content = MagicMock()
        evt.content.body = "!fsearch test query"

        # Mock Discourse API search results
        self.bot.discourse_api.search_discourse.return_value = [
            {
                'title': 'Test Topic 1',
                'slug': 'test-topic-1',
                'id': 1,
                'views': 100,
                'created_at': '2023-01-01T00:00:00Z'
            },
            {
                'title': 'Test Topic 2',
                'slug': 'test-topic-2',
                'id': 2,
                'views': 150,
                'created_at': '2023-01-02T00:00:00Z'
            },
        ]

        # Call the implementation directly
        await self.bot.handle_search_discourse(evt, query='test query')

        # Assert that reply was called
        evt.reply.assert_called()
        # Optionally, check the content of the reply

    async def test_search_discourse_no_results(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        evt.content = MagicMock()
        evt.content.body = "!fsearch noresults"

        # Mock Discourse API to return no results
        self.bot.discourse_api.search_discourse.return_value = []

        # Call the implementation directly
        await self.bot.handle_search_discourse(evt, query='noresults')

        # Assert that 'No results found.' message is sent
        evt.reply.assert_called_with("No results found.")

    async def test_process_link_success(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        message_body = 'Check this out: https://example.com/article'

        # Mock URL extraction
        with patch('url_handler.extract_urls', return_value=['https://example.com/article']):
            # Mock Discourse API check_for_duplicate
            self.bot.discourse_api.check_for_duplicate.return_value = False
            # Mock scrape_content
            with patch('url_handler.scrape_content', return_value='Scraped content'):
                # Mock summarization
                self.bot.ai_integration.summarize_content.return_value = 'Summarized content'
                # Mock generate_title
                self.bot.ai_integration.generate_title.return_value = 'Generated Title'
                # Mock create_post
                self.bot.discourse_api.create_post.return_value = ('https://discourse.example.com/t/test-topic/1', None)

                await self.bot.process_link(evt, message_body)
                # Assert that the post was created successfully
                evt.reply.assert_called_with('Post created successfully! URL: https://discourse.example.com/t/test-topic/1')

    async def test_process_link_duplicate(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        message_body = 'Check this out: https://example.com/article'

        # Mock URL extraction
        with patch('url_handler.extract_urls', return_value=['https://example.com/article']):
            # Mock Discourse API check_for_duplicate
            self.bot.discourse_api.check_for_duplicate.return_value = True

            await self.bot.process_link(evt, message_body)
            # Assert that a duplicate message was sent
            evt.reply.assert_called_with('A post with this URL already exists: https://example.com/article')

    async def test_post_to_discourse_discourse_api_failure(self):
        # Mock event
        evt = MagicMock(spec=MessageEvent)
        evt.reply = AsyncMock()
        evt.content = MagicMock()  # Ensure evt.content is initialized
        evt.content.get_reply_to.return_value = "event_id"
        evt.content.body = "!fpost"
        evt.client = AsyncMock()
        evt.client.get_event.return_value = MagicMock()
        evt.client.get_event.return_value.content.body = "This is the message body"
        evt.room_id = "!roomid:example.com"

        # Mock AI integration
        self.bot.ai_integration.generate_title.return_value = "Generated Title"
        # Mock Discourse API failure
        self.bot.discourse_api.create_post.return_value = (None, 'Failed to create post: Error message')

        # Call the implementation directly
        await self.bot.handle_post_to_discourse(evt)

        # Assert that an error message was sent
        evt.reply.assert_called_with('Failed to create post: Failed to create post: Error message')

    def test_extract_urls(self):
        text = 'Visit https://example.com and http://test.com for more info.'
        urls = extract_urls(text)
        self.assertEqual(urls, ['https://example.com', 'http://test.com'])

    def test_generate_bypass_links(self):
        url = 'https://example.com/article'
        links = generate_bypass_links(url)
        expected_links = {
            'original': 'https://example.com/article',
            '12ft': 'https://12ft.io/https://example.com/article',
            'archive': 'https://web.archive.org/web/https://example.com/article',
        }
        self.assertEqual(links, expected_links)

    async def test_create_post_success(self):
        # Mock DiscourseAPI instance
        discourse_api = DiscourseAPI(self.mock_config, self.mock_log)

        # Mock aiohttp ClientSession.post
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.__aenter__.return_value.status = 200
            mock_response.__aenter__.return_value.text.return_value = '{"topic_id": 1, "topic_slug": "test-topic"}'
            mock_response.__aenter__.return_value.json.return_value = {"topic_id": 1, "topic_slug": "test-topic"}
            mock_post.return_value = mock_response

            post_url, error = await discourse_api.create_post(
                title='Test Title',
                raw='Test body',
                category_id=1,
                tags=[]
            )
            self.assertEqual(post_url, 'https://discourse.example.com/t/test-topic/1')
            self.assertIsNone(error)

if __name__ == "__main__":
    unittest.main()