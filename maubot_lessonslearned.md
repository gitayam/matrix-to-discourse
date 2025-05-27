xxxmarkdown

# Lessons Learned - Efficient Maubot Plugin Development

This document captures key lessons learned during the development and maintenance of Maubot plugins. These insights aim to streamline future development, debugging, and ensure robust and maintainable bots, applicable for individual developers or small teams.

## Table of Contents

1.  [Maubot Plugin Structure and Lifecycle](https://www.google.com/search?q=%23maubot-plugin-structure-and-lifecycle)
2.  [Effective Configuration Management](https://www.google.com/search?q=%23effective-configuration-management)
3.  [Handling Matrix Events and Commands](https://www.google.com/search?q=%23handling-matrix-events-and-commands)
4.  [Asynchronous Programming Best Practices](https://www.google.com/search?q=%23asynchronous-programming-best-practices)
5.  [Database and State Management](https://www.google.com/search?q=%23database-and-state-management)
6.  [Error Handling and Logging](https://www.google.com/search?q=%23error-handling-and-logging)
7.  [Testing and Deployment](https://www.google.com/search?q=%23testing-and-deployment)
8.  [Standard Operating Procedures](https://www.google.com/search?q=%23standard-operating-procedures)

-----

## 1\. Maubot Plugin Structure and Lifecycle

### ❌ What Didn't Work

**Problem**: Misunderstanding `maubot.yaml`'s role or `main_class` resolution.

  - Plugin not loading due to incorrect `modules` or `main_class` paths.
  - Modules not being loaded in the correct order, leading to `ModuleNotFoundError`.

**Problem**: Ignoring plugin lifecycle methods (`start`, `stop`).

  - Resources (e.g., HTTP sessions, database connections) not being properly closed.
  - Long-running tasks not being gracefully shut down.

### ✅ What Worked

**Solution**: Strict adherence to `maubot.yaml` and proper use of lifecycle methods.

**`maubot.yaml` structure**:

  - `id`: Unique identifier (Java package naming recommended, e.g., `com.example.mybot`).
  - `version`: PEP 440 compliant.
  - `modules`: List all Python modules in order of dependency. The module containing `main_class` usually comes last.
  - `main_class`: `module_name.ClassName`.

Example `maubot.yaml`:
```
maubot: 0.1.0
id: com.example.helloworld
version: 1.0.0
license: MIT
modules:

  - helloworld
    main\_class: helloworld.HelloWorldBot
    extra\_files:
  - base-config.yaml \# If using configuration
    config: true \# Enable config if base-config.yaml is present
    ```

**Plugin Class Structure and Lifecycle**:

  - Inherit `maubot.Plugin`.
  - Use `async def start()` for initialization (e.g., loading config, setting up external connections).
  - Use `async def stop()` for cleanup (e.g., closing connections, cancelling tasks).
  - Use `def get_config_class()` if your plugin uses configuration.

Example Python (e.g., `helloworld/__init__.py`):
```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command
from maubot.config import Base and Dict

class Config(Base):
prefix: str = "\!"
welcome\_message: str = "Hello, Maubot\!"

class HelloWorldBot(Plugin):
config: Config \# Type hint for easier access

```
async def start(self) -> None:
    self.log.info("Hello World Bot starting...")
    self.config.load_and_update() # Load config on startup
    self.log.info(f"Using prefix: {self.config.prefix}")

async def stop(self) -> None:
    self.log.info("Hello World Bot stopping.")
    # No specific cleanup needed for this simple bot

def get_config_class(self) -> type[Config]:
    return Config

@command.new("hello")
async def hello_command(self, evt: MessageEvent) -> None:
    await evt.reply(f"{self.config.welcome_message} I am a Maubot plugin!")
```

```

### 🔧 Standard Operating Procedure

1.  **Always define a clear `maubot.yaml`** with correct `id`, `version`, `modules`, and `main_class`.
2.  **Explicitly list all top-level Python modules** in `maubot.yaml`'s `modules` section.
3.  **Implement `async def start()` for any setup tasks** that need to run when the plugin loads.
4.  **Implement `async def stop()` for graceful shutdown**, releasing resources like database connections or HTTP clients.
5.  **Use `self.config.load_and_update()` in `start()`** if your plugin has configuration.

-----

## 2\. Effective Configuration Management

### ❌ What Didn't Work

**Problem**: Hardcoding configurable values directly in the code.

  - Requires modifying and redeploying the plugin for simple changes.
  - Makes it difficult for users to customize the bot without touching code.

**Problem**: Inconsistent config schemas or migrations.

  - Breaking existing plugin instances when updating the plugin.
  - Loss of user-defined configuration values.

### ✅ What Worked

**Solution**: Leverage Maubot's built-in configuration system with `maubot.config.Base`.

1.  **Define a `Config` class inheriting from `maubot.config.Base`**: This allows Maubot's management UI to automatically generate a config editor.
2.  **Provide a `base-config.yaml`**: This file, bundled with your plugin (`extra_files` in `maubot.yaml`), provides default values and the initial schema.
3.  **Implement `get_config_class()`**: This method in your `Plugin` class tells Maubot which class to use for configuration.
4.  **Handle config updates with `do_update` (if schema changes)**: For schema migrations, override `do_update` in your `Config` class to safely migrate old configurations to new ones.
5.  **Access config via `self.config`**: Maubot automatically loads the configuration into `self.config` (typed by your `Config` class).

Example `base-config.yaml`:
```
prefix: "\!"
admin\_users: [] \# List of Matrix user IDs who are admins
response\_delay\_ms: 500
```

Example `Config` class with migration:
```python
from maubot.config import Base, ConfigUpdateHelper
from typing import Type, Optional

class OldConfig(Base): \# Represents an older config schema
prefix: str = "\!"
admin\_id: Optional[str] = None \# Old field, now a list

class Config(Base): \# Current config schema
prefix: str = "\!"
admin\_users: list[str] = [] \# New field, a list
response\_delay\_ms: int = 500

```
def do_update(self, helper: ConfigUpdateHelper) -> None:
    # Example migration from OldConfig to Config
    old_config = helper.as_dict(OldConfig)
    if old_config.get("admin_id") and not self.admin_users:
        self.admin_users = [old_config["admin_id"]]
    
    # Copy other fields if they exist in source but not in base (optional)
    helper.copy("prefix")
    helper.copy("response_delay_ms")
```

```

### 🔧 Standard Operating Procedure

1.  **Always use `maubot.config.Base`** for plugin configuration.
2.  **Define sensible defaults in `base-config.yaml`**.
3.  **Include `base-config.yaml` in `extra_files`** and set `config: true` in `maubot.yaml`.
4.  **Implement `get_config_class()`** in your main plugin class.
5.  **Plan for config schema changes**: Use `do_update` in your `Config` class to ensure smooth migrations between plugin versions.
6.  **Access configuration through `self.config`** in your plugin methods.

-----

## 3\. Handling Matrix Events and Commands

### ❌ What Didn't Work

**Problem**: Imperfect command parsing or event handling.

  - Bots responding to themselves or other bots.
  - Commands not triggering due to case sensitivity, extra spaces, or missing prefixes.
  - Not handling message types beyond text (e.g., edits, reactions).

### ✅ What Worked

**Solution**: Use Maubot's decorators and helper methods for robust handling.

  - **`@command.new()` for commands**: Maubot handles prefix matching, subcommands, and argument parsing.
      - Specify `prefix` in `maubot.yaml` or config.
      - Use `args` and `arg_fallthrough` for complex arguments.
      - Use `parser_class` for custom argument parsing (e.g., `mautrix.util.command.Args`).
  - **`@event.on()` for raw Matrix events**: For custom event types or more granular control.
  - **`MessageEvent` methods**: Use `evt.reply()`, `evt.respond()`, `evt.redact()`, `evt.react()` for common Matrix actions.
  - **Filtering events**: Use `evt.sender == self.client.mxid` to prevent the bot from responding to itself. Check `evt.room_id` for room-specific logic.

Example command and event handling:
```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event
from mautrix.types import EventType, RoomID, RelationType, RedactionEventContent

class MyBot(Plugin):
@command.new("echo", help="Echoes your message back.")
async def echo\_command(self, evt: MessageEvent, message: str) -\> None:
await evt.reply(message)

```
@command.new("react", help="Reacts to your message with a 👍.")
async def react_command(self, evt: MessageEvent) -> None:
    await evt.react("👍")

@command.new("delete_message", help="Redacts a message (requires permission).")
async def delete_message_command(self, evt: MessageEvent, target_event_id: str) -> None:
    if not self.is_admin(evt.sender): # Custom admin check
        await evt.reply("You are not an admin.")
        return
    
    redaction_content = RedactionEventContent(redacts=target_event_id, reason="Requested by admin")
    await self.client.send_event(evt.room_id, EventType.ROOM_REDACTION, redaction_content)
    await evt.reply(f"Attempted to redact event {target_event_id}")


@event.on(EventType.ROOM_MESSAGE)
async def on_message(self, evt: MessageEvent) -> None:
    if evt.sender == self.client.mxid: # Ignore own messages
        return
    
    if evt.content.msgtype == "m.text" and "hello" in evt.content.body.lower():
        await evt.reply("Hello there!")

    if evt.content.relates_to and evt.content.relates_to.rel_type == RelationType.REPLACE:
        self.log.info(f"Message {evt.event_id} was edited.")
        # Handle message edits here
```

```

### 🔧 Standard Operating Procedure

1.  **Use `@command.new()` for most bot interactions**. It handles common command parsing.
2.  **Use `evt.reply()` for direct replies** to the message that triggered the command.
3.  **For complex interactions or specific event types, use `@event.on()`**.
4.  **Always filter out your bot's own messages** (`evt.sender == self.client.mxid`) to prevent infinite loops.
5.  **Consider using `mbc build --upload`** for quick development iterations to see changes immediately.
6.  **Be mindful of permissions**: Ensure the bot account has the necessary Matrix power levels (e.g., `redact`, `kick`, `ban`) for administrative commands.

-----

## 4\. Asynchronous Programming Best Practices

### ❌ What Didn't Work

**Problem**: Blocking the event loop with synchronous operations.

  - Maubot becoming unresponsive.
  - Delays in processing other events.

**Problem**: Not managing async tasks properly.

  - Tasks running indefinitely after plugin stops.
  - Unhandled exceptions in background tasks.

### ✅ What Worked

**Solution**: Embrace `asyncio` and `aiohttp`.

  - **Always `await` I/O operations**: Network requests, database queries, file I/O must be `await`ed. Use `self.client` for Matrix API calls and `self.http` for external HTTP requests.
  - **Use `asyncio.create_task` for background tasks**: If a task doesn't need to block the current handler, run it in the background.
  - **Manage background tasks in `start()` and `stop()`**: Store references to tasks created with `asyncio.create_task` and cancel them in `stop()`.

Example async operations:
```python
import asyncio
import aiohttp

class MyBot(Plugin):
\_periodic\_task: Optional[asyncio.Task] = None

```
async def start(self) -> None:
    self.log.info("Starting async operations...")
    # Start a periodic task
    self._periodic_task = asyncio.create_task(self._send_periodic_message())

async def stop(self) -> None:
    self.log.info("Stopping async operations...")
    if self._periodic_task:
        self._periodic_task.cancel()
        try:
            await self._periodic_task
        except asyncio.CancelledError:
            self.log.info("Periodic task cancelled.")

async def _send_periodic_message(self) -> None:
    try:
        while True:
            # Example: Fetch data from an external API
            async with self.http.get("[https://api.example.com/status](https://api.example.com/status)") as resp:
                status_data = await resp.json()
                self.log.info(f"API status: {status_data.get('status')}")

            # Example: Send a message to a specific room
            # Replace with a real room ID for testing
            # await self.client.send_text(RoomID("!some_room:example.com"), "Still alive!")

            await asyncio.sleep(60) # Wait for 60 seconds
    except asyncio.CancelledError:
        self.log.warning("Periodic message task was cancelled.")
    except aiohttp.ClientError as e:
        self.log.error(f"HTTP client error in periodic task: {e}")
    except Exception as e:
        self.log.exception(f"Unhandled exception in periodic task: {e}")

@command.new("fetch_external_data")
async def fetch_data_command(self, evt: MessageEvent) -> None:
    try:
        async with self.http.get("[https://api.example.com/data](https://api.example.com/data)") as resp:
            resp.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = await resp.json()
            await evt.reply(f"Fetched data: {data}")
    except aiohttp.ClientResponseError as e:
        await evt.reply(f"Error fetching data: {e.status} {e.message}")
    except aiohttp.ClientError as e:
        await evt.reply(f"Network error fetching data: {e}")
    except Exception as e:
        self.log.exception("Unexpected error in fetch_data_command")
        await evt.reply("An unexpected error occurred while fetching data.")
```

```

### 🔧 Standard Operating Procedure

1.  **Always use `await` for any I/O-bound operations** (Matrix API, HTTP, database).
2.  **Avoid `time.sleep()` or synchronous blocking calls**; use `asyncio.sleep()`.
3.  **For background tasks, use `asyncio.create_task()`** and store the task reference.
4.  **Implement proper cancellation logic in `stop()`** for all background tasks.
5.  **Use `aiohttp` for external HTTP requests** through `self.http`.

-----

## 5\. Database and State Management

### ❌ What Didn't Work

**Problem**: Storing transient state in global variables.

  - State is lost on plugin reload.
  - Difficult to scale or manage across multiple instances.

**Problem**: Direct SQLite usage without proper abstractions.

  - Boilerplate SQL queries everywhere.
  - Potential for SQL injection (though Maubot uses `aiosqlite` which helps).
  - No schema migration strategy.

### ✅ What Worked

**Solution**: Use Maubot's built-in database (SQLite or PostgreSQL) and `mautrix.util.async_db.Database` (or `SQLAlchemy` for complex needs).

  - **Enable database in `maubot.yaml`**: Set `database: true`.
  - **Access via `self.database`**: Your plugin gets a database connection.
  - **Use `mautrix.util.async_db.Database`**: This provides basic async database operations.
  - **Schema migrations**: Maubot encourages simple schema versioning using `CREATE TABLE IF NOT EXISTS` and checking for column existence, or more robust tools like Alembic for complex projects.
  - **Persist critical state in the database**: User-specific settings, counters, or application state.

Example database usage:
```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command
from typing import Optional

class DataBot(Plugin):
async def start(self) -\> None:
await super().start()
\# Initialize database schema if it doesn't exist
await self.database.execute('''
CREATE TABLE IF NOT EXISTS user\_counters (
user\_id TEXT PRIMARY KEY,
count INTEGER DEFAULT 0
)
''')
self.log.info("Database schema checked/initialized.")

```
@command.new("increment")
async def increment_command(self, evt: MessageEvent) -> None:
    user_id = evt.sender.to_string()

    async with self.database.transaction():
        # Fetch current count
        row = await self.database.fetchrow(
            "SELECT count FROM user_counters WHERE user_id = $1", user_id
        )
        current_count = row["count"] if row else 0

        new_count = current_count + 1

        # Update or insert
        await self.database.execute(
            """
            INSERT INTO user_counters (user_id, count)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET count = $2
            """,
            user_id,
            new_count,
        )
    await evt.reply(f"Your count is now: {new_count}")

@command.new("my_count")
async def my_count_command(self, evt: MessageEvent) -> None:
    user_id = evt.sender.to_string()
    row = await self.database.fetchrow(
        "SELECT count FROM user_counters WHERE user_id = $1", user_id
    )
    current_count = row["count"] if row else 0
    await evt.reply(f"Your current count is: {current_count}")
```

```

### 🔧 Standard Operating Procedure

1.  **Enable database access in `maubot.yaml`** (`database: true`).
2.  **Perform schema initialization/migrations in `start()`** using `self.database.execute()`.
3.  **Use `self.database.fetchrow()` or `Workspace()`** for querying data.
4.  **Use `self.database.execute()`** for inserts, updates, and deletions.
5.  **Always use parameterized queries (`$1`, `$2`, etc.)** to prevent SQL injection.
6.  **Use `async with self.database.transaction():`** for atomic database operations.
7.  **Avoid direct file I/O for persistent state** unless absolutely necessary; prefer the database.

-----

## 6\. Error Handling and Logging

### ❌ What Didn't Work

**Problem**: Silent failures or generic error messages.

  - Difficult to diagnose issues without proper context.
  - Frustrates users when the bot doesn't explain what went wrong.

**Problem**: Over-reliance on `print()` for debugging.

  - Unstructured output, hard to filter.
  - Not visible in Maubot's integrated logs.

### ✅ What Worked

**Solution**: Utilize `self.log` and comprehensive `try...except` blocks.

  - **Use `self.log` for all logging**: Maubot provides a pre-configured `logging.Logger` instance.
      - `self.log.debug()`: Detailed information for debugging.
      - `self.log.info()`: General progress/status messages.
      - `self.log.warning()`: Something unexpected happened, but operation can continue.
      - `self.log.error()`: An error occurred, but the bot might recover.
      - `self.log.exception()`: Use inside an `except` block to log the traceback.
  - **Specific exception handling**: Catch specific exceptions (e.g., `aiohttp.ClientError`, `asyncio.CancelledError`) rather than a broad `except Exception`.
  - **Informative error messages to users**: Reply to the user with a helpful message when an error occurs, if appropriate.

Example error handling and logging:
```python
import aiohttp

class RobustBot(Plugin):
@command.new("fetch\_failing\_data")
async def fetch\_failing\_data\_command(self, evt: MessageEvent) -\> None:
try:
async with self.http.get("[https://api.example.com/nonexistent\_endpoint](https://www.google.com/search?q=https://api.example.com/nonexistent_endpoint)") as resp:
resp.raise\_for\_status() \# Raises aiohttp.ClientResponseError for 4xx/5xx
data = await resp.json()
await evt.reply(f"Fetched data: {data}")
except aiohttp.ClientResponseError as e:
self.log.error(f"API returned error: {e.status} - {e.message} for {e.request\_info.url}")
await evt.reply(f"Error from API: {e.status}. Please try again later.")
except aiohttp.ClientConnectorError as e:
self.log.exception(f"Network connection error to API: {e}")
await evt.reply("Could not connect to the external API. Please check network.")
except asyncio.TimeoutError:
self.log.warning("External API request timed out.")
await evt.reply("The request to the external API timed out.")
except Exception as e:
self.log.exception(f"An unexpected error occurred in fetch\_failing\_data\_command: {e}")
await evt.reply("An unexpected error occurred. The bot maintainer has been notified.")
```

### 🔧 Standard Operating Procedure

1.  **Use `self.log` for all output**: Match log levels to the severity and nature of the message.
2.  **Employ specific `try...except` blocks** to handle anticipated errors.
3.  **Use `self.log.exception()` within `except` blocks** to automatically include stack traces.
4.  **Provide clear, user-friendly error messages** in replies, while logging more technical details for debugging.
5.  **Monitor Maubot's logs** (via the web UI or CLI) regularly during development and deployment.

-----

## 7\. Testing and Deployment

### ❌ What Didn't Work

**Problem**: Relying solely on manual testing in a live Matrix room.

  - Slow, repetitive, and error-prone.
  - Difficult to test edge cases or error conditions.

**Problem**: Manual plugin building and uploading.

  - Prone to mistakes, forgetting steps.

### ✅ What Worked

**Solution**: Automate testing and streamline deployment with `mbc`.

  - **Unit Testing**: Write Python unit tests for individual functions and classes in your plugin logic. Mock `self.client`, `self.http`, and `self.database` for isolated testing.
  - **Integration Testing**: Use a dedicated test Matrix room and a separate Maubot instance for integration tests. Manually (or automate) test bot commands and reactions.
  - **`mbc` CLI Tool**:
      - `mbc build`: Packages your plugin into a `.mbp` file.
      - `mbc build --upload`: Builds and directly uploads to your Maubot instance (if `mbc login` is configured). This is crucial for rapid development.
      - `mbc logs`: View plugin logs from the CLI.
      - `mbc auth`: Obtain access tokens for bot accounts.
  - **Virtual Environments**: Always develop within a Python virtual environment (`venv`) to manage dependencies.

Example `pyproject.toml` (for Poetry, similar for `requirements.txt`):
```toml
[tool.poetry]
name = "my-maubot-plugin"
version = "1.0.0"
description = "A Maubot plugin example"
authors = ["Your Name [you@example.com](mailto:you@example.com)"]

[tool.poetry.dependencies]
python = "\>=3.10,\<4.0"
maubot = "^0.5.0" \# Ensure compatible Maubot version

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0" \# For async tests
pytest-mock = "^3.12.0" \# For mocking maubot components

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

Example basic test (e.g., `tests/test_myplugin.py`):
```python
import pytest
from unittest.mock import AsyncMock
from maubot.client import MaubotClient
from maubot.matrix import MessageEvent
from myplugin.main\_module import MyBot \# Assuming your main class is MyBot in main\_module.py

@pytest.fixture
def mock\_plugin():
\# Mocking Maubot's Plugin class and its properties
plugin = MyBot()
plugin.client = AsyncMock(spec=MaubotClient)
plugin.log = AsyncMock() \# Mock the logger
plugin.config = AsyncMock() \# Mock the config
plugin.config.prefix = "\!" \# Set a default prefix for testing
return plugin

@pytest.mark.asyncio
async def test\_echo\_command(mock\_plugin):
evt = AsyncMock(spec=MessageEvent)
evt.sender.to\_string.return\_value = "@user:example.com"
evt.room\_id = "\!room:example.com"
evt.content.body = "\!echo test message"

```
# Manually call the command handler
await mock_plugin.echo_command(evt, "test message")

mock_plugin.client.send_message.assert_awaited_once_with(
    evt.room_id, "test message",
    in_reply_to=evt.event_id
)
```

@pytest.mark.asyncio
async def test\_admin\_check\_for\_delete\_message(mock\_plugin):
evt = AsyncMock(spec=MessageEvent)
evt.sender.to\_string.return\_value = "@non\_admin:example.com"

```
# Simulate a command that requires admin
mock_plugin.is_admin.return_value = False # Assuming you have an is_admin helper
await mock_plugin.delete_message_command(evt, "$some_event_id")

mock_plugin.client.send_message.assert_awaited_once_with(
    evt.room_id, "You are not an admin.",
    in_reply_to=evt.event_id
)
mock_plugin.client.send_event.assert_not_awaited() # Ensure redact wasn't called
```

```

### 🔧 Standard Operating Procedure

1.  **Develop in a Python virtual environment.**
2.  **Write unit tests** for your plugin's core logic, mocking Maubot dependencies. Use `pytest` and `pytest-asyncio`.
3.  **Use `mbc build` to generate `.mbp` files** for release.
4.  **During active development, use `mbc build --upload`** for quick iterations.
5.  **Test plugins in a dedicated test Matrix room** with a separate bot account/instance to avoid affecting production.
6.  **Familiarize yourself with Maubot's web management UI** for uploading, creating instances, and viewing logs in a production environment.

-----

## 8\. Standard Operating Procedures

### Plugin Development Workflow

1.  **Define plugin purpose**: What problem does it solve or what feature does it add?
2.  **Design `maubot.yaml`**: ID, version, modules, main class.
3.  **Define `Config` class and `base-config.yaml`**: Set up defaults and structure.
4.  **Implement core logic**: Use Maubot's handlers (`@command`, `@event.on`).
5.  **Integrate external services**: Use `self.http` with `asyncio`.
6.  **Manage persistent state**: Use `self.database` for data storage and schema management.
7.  **Add comprehensive logging**: Use `self.log` at appropriate levels.
8.  **Write unit tests**: Cover core logic, edge cases, and error conditions.
9.  **Iterate quickly**: Use `mbc build --upload` and test in a dev Matrix room.
10. **Refine and Document**: Clean up code, add comments, update documentation for future use.

### Debugging Workflow

1.  **Check Maubot UI logs first**: The fastest way to see immediate errors.
2.  **Use `mbc logs <plugin_id>`**: For more detailed CLI access to logs.
3.  **Add `self.log.debug()` statements**: Strategically place them around suspicious code.
4.  **Verify `maubot.yaml` and `base-config.yaml`**: Often, simple typos or incorrect paths here cause load failures.
5.  **Isolate the problem**: Temporarily comment out parts of the code to narrow down the source of the issue.
6.  **Run unit tests**: See if the issue is caught by existing tests, or write a new test to reproduce it.
7.  **Consult Maubot documentation and Matrix dev rooms**: For obscure errors, the Maubot docs or the `#maubot:maunium.net` Matrix room are excellent resources.

### Code Quality Checklist

  - [ ] `maubot.yaml` is correctly defined and up-to-date.
  - [ ] Plugin adheres to `maubot.config.Base` for configuration.
  - [ ] `async def start()` and `async def stop()` are implemented for resource management.
  - [ ] All I/O operations are `await`ed.
  - [ ] Background tasks are managed using `asyncio.create_task` and properly cancelled.
  - [ ] `self.log` is used consistently for all output.
  - [ ] `try...except` blocks are specific and handle potential errors gracefully.
  - [ ] Database access uses `self.database` and parameterized queries.
  - [ ] Unit tests exist for critical components.
  - [ ] Bot accounts have necessary Matrix power levels for administrative commands.
  - [ ] Sensitive information (tokens, passwords) is never hardcoded and handled via configuration.

### Testing Strategy

1.  **Unit Testing**:
      - Test individual functions/methods in isolation.
      - Mock `self.client`, `self.http`, `self.database` to decouple tests.
      - Test happy paths, edge cases, and error conditions.
2.  **Integration Testing**:
      - Deploy plugin to a dedicated non-production Maubot instance.
      - Test all commands and event handlers in a real Matrix room.
      - Verify interactions with external APIs or databases.
3.  **Manual Testing**:
      - User acceptance testing (UAT) in a real Matrix environment.
      - Test with different Matrix clients (Element, Cinny, etc.).
      - Test with different configurations.

-----

## Key Takeaways

1.  **Maubot is asynchronous**: Embrace `asyncio` and `await` everything that can block the event loop.
2.  **Configuration is king**: Use `maubot.config.Base` and `base-config.yaml` for flexibility and easy management.
3.  **Leverage Maubot's abstractions**: Use `self.client`, `self.http`, `self.database`, and command/event decorators.
4.  **Robust error handling is crucial**: Log comprehensively with `self.log.exception()` and provide user-friendly messages.
5.  **Automate deployment with `mbc`**: `mbc build --upload` is a time-saver.
6.  **Test early and often**: Unit tests prevent regressions, integration tests ensure real-world functionality.
7.  **Understand the plugin lifecycle**: Initialize in `start()`, clean up in `stop()`.
8.  **Security matters**: Never hardcode credentials; validate user input.

This document should be updated as new lessons are learned during continued Maubot plugin development.
```