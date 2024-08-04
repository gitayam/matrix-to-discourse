import sys
import os
from unittest.mock import Mock

# Add the path to the module to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'matrix_to_discourse')))

from matrix_to_discourse import MatrixToDiscourseBot

def test_bot():
    try:
        # Create mock objects for required parameters
        mock_client = Mock()
        mock_loop = Mock()
        mock_http = Mock()
        mock_instance_id = "test_instance_id"
        mock_log = Mock()
        mock_config = {"discourse_api_key": "test_key", "discourse_api_username": "test_user", "discourse_base_url": "https://forum.irregularchat.com"}
        mock_database = Mock()
        mock_webapp = Mock()
        mock_webapp_url = "https://webapp.url"
        mock_loader = Mock()

        # Instantiate the bot with mock parameters
        bot = MatrixToDiscourseBot(
            client=mock_client,
            loop=mock_loop,
            http=mock_http,
            instance_id=mock_instance_id,
            log=mock_log,
            config=mock_config,
            database=mock_database,
            webapp=mock_webapp,
            webapp_url=mock_webapp_url,
            loader=mock_loader
        )

        print("MatrixToDiscourseBot imported and instantiated successfully.")
    except Exception as e:
        print("Error importing or instantiating MatrixToDiscourseBot:", e)

if __name__ == "__main__":
    test_bot()
