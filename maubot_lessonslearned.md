# Lessons Learned - Efficient Maubot Plugin Development

This document captures key lessons learned during the development and maintenance of Maubot plugins, based on analysis of real-world plugin implementations including reminder, github, faqbot, reactbot, twitchbot, and timer plugins. These insights aim to streamline future development, debugging, and ensure robust and maintainable bots, applicable for individual developers or small teams.

## Table of Contents

1.  [Maubot Plugin Structure and Lifecycle](#maubot-plugin-structure-and-lifecycle)
2.  [Effective Configuration Management](#effective-configuration-management)
3.  [Handling Matrix Events and Commands](#handling-matrix-events-and-commands)
4.  [Asynchronous Programming Best Practices](#asynchronous-programming-best-practices)
5.  [Database and State Management](#database-and-state-management)
6.  [Performance Optimization and Security Hardening](#performance-optimization-and-security-hardening)
7.  [Error Handling and Logging](#error-handling-and-logging)
8.  [Testing and Deployment](#testing-and-deployment)
9.  [Plugin Architecture Patterns](#plugin-architecture-patterns)
10. [Multi-File Plugin Organization](#multi-file-plugin-organization)
11. [Service Integration and Inter-Service Communication](#service-integration-and-inter-service-communication)
12. [Matrix Message Handling, Encryption, and Security](#matrix-message-handling-encryption-and-security)
13. [Text Transformation and Replacement Patterns](#text-transformation-and-replacement-patterns)
14. [Standard Operating Procedures](#standard-operating-procedures)

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

  - `id`: Unique identifier (reverse domain naming recommended, e.g., `org.bytemarx.reminder`, `xyz.maubot.github`).
  - `version`: Semantic versioning (e.g., `0.1.2`, `2.2.0`).
  - `modules`: List all Python modules. For single-file plugins, just the filename without `.py`.
  - `main_class`: `ClassName` for single-file plugins, or `module_name.ClassName` for multi-module.
  - `license`: Consistently use `AGPL-3.0-or-later` (observed standard across plugins).
  - `database`: Set to `true` if plugin needs database access.
  - `database_type`: Use `asyncpg` for PostgreSQL compatibility.
  - `webapp`: Set to `true` if plugin provides web endpoints.
  - `config`: Set to `true` if plugin uses configuration.

**Real-world examples from analyzed plugins**:

Single-file plugin (`timer`):
```yaml
maubot: 0.1.0
id: mx.quinn.timer
version: 1.0.0
license: MIT
modules:
  - timer
main_class: Timer
```

Multi-module plugin (`reminder`):
```yaml
maubot: 0.4.1
id: org.bytemarx.reminder
version: 0.1.2
license: AGPL-3.0-or-later
modules:
- reminder
main_class: ReminderBot
extra_files:
- base-config.yaml
dependencies:
- pytz
- dateparser
- apscheduler
soft_dependencies:
- cron_descriptor
database: true
database_type: asyncpg
```

Complex plugin with webapp (`github`):
```yaml
maubot: 0.4.1
id: xyz.maubot.github
version: 0.2.0
license: AGPL-3.0-or-later
modules:
- github
main_class: GitHubBot
extra_files:
- base-config.yaml
database: true
database_type: asyncpg
webapp: true
config: true
```

**Plugin Class Structure and Lifecycle**:

  - Inherit `maubot.Plugin`.
  - Use `async def start()` for initialization (e.g., loading config, setting up external connections, database setup).
  - Use `async def stop()` for cleanup (e.g., closing connections, cancelling tasks).
  - Use `@classmethod def get_config_class()` if your plugin uses configuration.
  - Use `@classmethod def get_db_upgrade_table()` for database migrations.

**Real-world patterns observed**:

Simple single-file plugin (`timer`):
```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command
import asyncio

class Timer(Plugin):
    async def start(self) -> None:
        self.namei = 0

    @command.new()
    @command.argument("time", required=True)
    @command.argument("name", required=False)
    async def timer(self, evt: MessageEvent, time: str, name: str | None) -> None:
        try:
            time = int(time)
        except ValueError:
            return await evt.reply(f"Invalid time '{time}' - try giving me a number of seconds")
        name = name or self.new_name()
        await evt.respond(f"⏳ Timer {name} started for {time} seconds")
        await asyncio.sleep(time)
        await evt.respond(f"⏰ Timer {name}: time's up!")
```

Complex multi-module plugin (`reminder`):
```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event
from mautrix.util.async_db import UpgradeTable
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        helper.copy("default_timezone")
        helper.copy("default_locale")
        helper.copy("base_command")

class ReminderBot(Plugin):
    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    @classmethod
    def get_db_upgrade_table(cls) -> UpgradeTable:
        return upgrade_table

    async def start(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
        self.db = ReminderDatabase(self.database)
        self.on_external_config_update()
        self.reminders = await self.db.load_all(self)

    async def stop(self) -> None:
        self.scheduler.shutdown(wait=False)

    def on_external_config_update(self) -> None:
        self.config.load_and_update()
        # Handle config changes...
```

Plugin with background tasks (`twitchbot`):
```python
class TwitchBot(Plugin):
    access_check_loop: Task

    async def start(self) -> None:
        self.config.load_and_update()
        self.access_check_loop = background_task.create(self.check_access_token())
        background_task.create(self.register_notifications())

    async def stop(self) -> None:
        self.access_check_loop.cancel()
```

### 🔧 Standard Operating Procedure

1.  **Always define a clear `maubot.yaml`** with correct `id`, `version`, `modules`, and `main_class`.
2.  **Use reverse domain naming for plugin IDs** (e.g., `org.example.pluginname`).
3.  **Set `database: true` and `database_type: asyncpg`** for database-enabled plugins.
4.  **Include `extra_files` for `base-config.yaml`** and set `config: true` for configurable plugins.
5.  **Implement `async def start()` for initialization** including config loading, database setup, and background tasks.
6.  **Implement `async def stop()` for graceful shutdown**, cancelling background tasks and releasing resources.
7.  **Use `@classmethod` decorators** for `get_config_class()` and `get_db_upgrade_table()`.
8.  **Follow the observed file structure patterns** based on plugin complexity.

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

**Solution**: Use `BaseProxyConfig` for robust configuration management (observed standard across plugins).

1.  **Define a `Config` class inheriting from `BaseProxyConfig`**: This is the standard pattern used by all analyzed plugins.
2.  **Provide a `base-config.yaml`**: This file, bundled with your plugin (`extra_files` in `maubot.yaml`), provides default values.
3.  **Implement `@classmethod get_config_class()`**: This method in your `Plugin` class tells Maubot which class to use for configuration.
4.  **Handle config updates with `do_update`**: Override `do_update` in your `Config` class using `ConfigUpdateHelper` for safe migrations.
5.  **Access config via `self.config`**: Maubot automatically loads the configuration.
6.  **Implement `on_external_config_update()`**: Handle runtime config changes without restart.

**Real-world configuration examples**:

Simple config (`faqbot`):
```yaml
# Who is allowed to save FAQ entries?
adminlist:
  - "@user:example.com"
# The prefix for the main command without the !
command_prefix: faq
# Whether typing "FAQ <term>" should retrieve the entry
FAQ_shortcut: true
```

Complex config (`reminder`):
```yaml
# Default timezone for users who did not set one.
default_timezone: America/Los_Angeles
# Default locale/language compatible with dateparser
default_locale: en
# Base command without the prefix (!).
base_command:
- remind
- remindme
# Agenda items are like reminders but don't have a time
agenda_command:
- agenda
- todo
# Rate limit for individual users
rate_limit: 10
rate_limit_minutes: 60
# Power level needed to delete someone else's reminder
admin_power_level: 50
```

**Config class patterns observed**:
```python
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        helper.copy("default_timezone")
        helper.copy("default_locale") 
        helper.copy("base_command")
        helper.copy("agenda_command")
        helper.copy("cancel_command")
        helper.copy("rate_limit_minutes")
        helper.copy("rate_limit")
        helper.copy("verbose")
        helper.copy("admin_power_level")
        helper.copy("time_format")
```

### 🔧 Standard Operating Procedure

1.  **Always use `BaseProxyConfig`** for plugin configuration (observed standard).
2.  **Define sensible defaults in `base-config.yaml`** with clear comments.
3.  **Include `base-config.yaml` in `extra_files`** and set `config: true` in `maubot.yaml`.
4.  **Implement `@classmethod get_config_class()`** in your main plugin class.
5.  **Use `helper.copy()` in `do_update()`** for each configuration field to ensure proper migration.
6.  **Implement `on_external_config_update()`** to handle runtime config changes.
7.  **Access configuration through `self.config`** and call `self.config.load_and_update()` in `start()`.

-----

## 3\. Handling Matrix Events and Commands

### ❌ What Didn't Work

**Problem**: Imperfect command parsing or event handling.

  - Bots responding to themselves or other bots.
  - Commands not triggering due to case sensitivity, extra spaces, or missing prefixes.
  - Not handling message types beyond text (e.g., edits, reactions).

### ✅ What Worked

**Solution**: Use Maubot's decorators and helper methods for robust handling (patterns from real plugins).

  - **`@command.new()` for commands**: Standard pattern across all plugins.
      - Use `@command.argument()` for parameter parsing with type hints.
      - Use `require_subcommand=False` and `arg_fallthrough=False` for flexible parsing.
      - Use lambda functions for dynamic command names: `name=lambda self: self.base_command[0]`.
  - **Subcommands**: Use `@main_command.subcommand()` for complex command hierarchies.
  - **`@command.passive()` for pattern matching**: Use regex patterns to catch specific message formats.
  - **`@event.on()` for Matrix events**: Handle tombstones, reactions, redactions.
  - **`@web.post()` for webhooks**: For external integrations (github, twitch).
  - **Event filtering**: Always check `evt.sender == self.client.mxid` to prevent self-responses.

**Real-world command patterns observed**:

Dynamic command names (`reminder`):
```python
@command.new(name=lambda self: self.base_command[0],
             aliases=lambda self, alias: alias in self.base_command + self.agenda_command,
             help="Create a reminder", require_subcommand=False, arg_fallthrough=False)
@command.argument("room", matches="room", required=False)
@command.argument("every", matches="every", required=False)
@command.argument("start_time", matches="(.*?);", pass_raw=True, required=False)
@command.argument("cron", matches="cron ?(?:\s*\S*){0,5}", pass_raw=True, required=False)
@command.argument("message", pass_raw=True, required=False)
async def create_reminder(self, evt: MessageEvent, room: str = None, every: str = None, 
                         start_time: Tuple[str] = None, cron: Tuple[str] = None, 
                         message: str = None, again: bool = False) -> None:
```

Subcommands (`faqbot`):
```python
@command.new(name=get_command_name)
async def faq(self, evt: MessageEvent) -> None:
    """Main command, nothing to do, this will call the help page"""
    pass

@faq.subcommand(help="Store a FAQ entry for this room")
@command.argument("key")
@command.argument("value", pass_raw=True)
async def save(self, evt: MessageEvent, key: str, value: str) -> None:
    if evt.sender not in self.config["adminlist"]:
        await evt.reply("You're not allowed to save FAQ entries.")
        return
    # ... save logic
```

Passive pattern matching (`faqbot`):
```python
@command.passive('^FAQ (\S*)', case_insensitive=True)
async def find_faq(self, evt: MessageEvent, match: Tuple[str]) -> None:
    """SHORTCUT to get FAQ entry"""
    if not self.config["FAQ_shortcut"]:
        return
    await self._get_faq(evt, match[1])
```

Reaction handling (`reminder`):
```python
@command.passive(regex=r"(?:\U0001F44D[\U0001F3FB-\U0001F3FF]?)",
                 field=lambda evt: evt.content.relates_to.key,
                 event_type=EventType.REACTION, msgtypes=None)
async def subscribe_react(self, evt: ReactionEvent, _: Tuple[str]) -> None:
    # Handle thumbs up reactions to subscribe to reminders
```

Webhook endpoints (`twitchbot`):
```python
@web.post("/stream-notify")
async def stream_notify(self, req: Request) -> Response:
    # Handle Twitch webhook notifications
```

Event handling (`faqbot`):
```python
@event.on(EventType.ROOM_TOMBSTONE)
async def tombstone(self, evt: StateEvent) -> None:
    """On room upgrades, also upgrade FAQ entries"""
    if not evt.content.replacement_room:
        return
    # Migrate data to new room
```

### 🔧 Standard Operating Procedure

1.  **Use `@command.new()` with `@command.argument()`** for structured command parsing.
2.  **Use lambda functions for dynamic command names** from config (e.g., `name=lambda self: self.config["command_prefix"]`).
3.  **Implement subcommands with `@main_command.subcommand()`** for complex functionality.
4.  **Use `@command.passive()` for pattern matching** instead of manual message parsing.
5.  **Handle Matrix events with `@event.on()`** for tombstones, reactions, redactions.
6.  **Use `@web.post()` for webhook endpoints** when integrating with external services.
7.  **Always filter out bot's own messages** (`evt.sender == self.client.mxid`).
8.  **Check user permissions** before executing admin commands (use config adminlists).

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

**Solution**: Embrace `asyncio` and use `mautrix.util.background_task` (observed pattern).

  - **Always `await` I/O operations**: Network requests, database queries, file I/O must be `await`ed. Use `self.client` for Matrix API calls and `self.http` for external HTTP requests.
  - **Use `background_task.create()` for background tasks**: This is the standard pattern used across plugins instead of `asyncio.create_task`.
  - **Manage background tasks in `start()` and `stop()`**: Store references to tasks and cancel them in `stop()`.
  - **Use schedulers for recurring tasks**: `AsyncIOScheduler` from `apscheduler` for complex scheduling needs.

**Real-world async patterns observed**:

Background tasks (`twitchbot`):
```python
from mautrix.util import background_task
from asyncio import Task, CancelledError, sleep

class TwitchBot(Plugin):
    access_check_loop: Task

    async def start(self) -> None:
        self.config.load_and_update()
        self.access_check_loop = background_task.create(self.check_access_token())
        background_task.create(self.register_notifications())

    async def stop(self) -> None:
        self.access_check_loop.cancel()

    async def check_access_token(self) -> None:
        try:
            while True:
                # Check token validity
                await sleep(3600)  # Sleep for an hour
        except CancelledError:
            self.log.debug("Access check loop stopped.")
```

Scheduler usage (`reminder`):
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class ReminderBot(Plugin):
    scheduler: AsyncIOScheduler

    async def start(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
        # Load and schedule existing reminders
        self.reminders = await self.db.load_all(self)

    async def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
```

HTTP requests with error handling (`twitchbot`):
```python
async def twitch_api(self, method: str, endpoint: str, params: dict | None, 
                     data: dict | None, evt: MessageEvent | None) -> dict | None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Client-Id": self.config["client_id"]
    }
    
    async with self.http.get("https://api.twitch.tv/helix" + endpoint, 
                            headers=headers, params=params) as response:
        if response.status == 401:
            # Handle token refresh
            access_token = await self.get_access_token()
        elif response.status >= 400:
            self.log.error(f"{method} {response.url}: {response.status}")
            return None
        return await response.json()
```

### 🔧 Standard Operating Procedure

1.  **Always use `await` for any I/O-bound operations** (Matrix API, HTTP, database).
2.  **Avoid `time.sleep()` or synchronous blocking calls**; use `asyncio.sleep()`.
3.  **Use `background_task.create()` for background tasks** (mautrix standard pattern).
4.  **Store task references and cancel them in `stop()`** for proper cleanup.
5.  **Use `AsyncIOScheduler` for complex scheduling** needs (reminders, periodic tasks).
6.  **Use `self.http` for external HTTP requests** with proper error handling.
7.  **Handle `CancelledError` in background tasks** for graceful shutdown.

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

**Solution**: Use Maubot's built-in database with `UpgradeTable` for migrations (observed standard).

  - **Enable database in `maubot.yaml`**: Set `database: true` and `database_type: asyncpg`.
  - **Access via `self.database`**: Your plugin gets a database connection.
  - **Use `UpgradeTable` for migrations**: This is the standard pattern across all database-enabled plugins.
  - **Implement `@classmethod get_db_upgrade_table()`**: Return your upgrade table for automatic migrations.
  - **Use parameterized queries**: Always use `$1`, `$2`, etc. for SQL parameters.
  - **Separate database logic**: Create dedicated DB classes for complex plugins.

**Real-world database patterns observed**:

Migration setup (`faqbot`):
```python
from mautrix.util.async_db import UpgradeTable, Connection

upgrade_table = UpgradeTable()

@upgrade_table.register(description="Initial revision")
async def upgrade_v1(conn: Connection) -> None:
    await conn.execute(
        """CREATE TABLE entries (
            roomid   TEXT,
            key   TEXT NOT NULL,
            value TEXT NOT NULL,
            UNIQUE(roomid, key) ON CONFLICT REPLACE
        )"""
    )

@upgrade_table.register(description="Make any roomid, key combo unique")
async def upgrade_v2(conn: Connection) -> None:
    await conn.execute("ALTER TABLE entries ADD COLUMN global INTEGER DEFAULT(0)")
    await conn.execute("CREATE INDEX global_entries ON entries(global)")
    await conn.execute("CREATE UNIQUE INDEX room_entries ON entries(roomid, key)")

class FaqBot(Plugin):
    @classmethod
    def get_db_upgrade_table(cls) -> UpgradeTable | None:
        return upgrade_table
```

Dedicated database class (`reminder`):
```python
class ReminderDatabase:
    def __init__(self, database):
        self.db = database
        self.defaults = UserDefaults()

    async def store_reminder(self, reminder: Reminder) -> None:
        await self.db.execute(
            "INSERT INTO reminders (id, room_id, message, creator, start_time) "
            "VALUES ($1, $2, $3, $4, $5)",
            reminder.id, reminder.room_id, reminder.message, 
            reminder.creator, reminder.start_time
        )

    async def load_all(self, bot) -> Dict[EventID, Reminder]:
        rows = await self.db.fetch("SELECT * FROM reminders")
        reminders = {}
        for row in rows:
            reminder = Reminder.from_row(row, bot)
            reminders[reminder.id] = reminder
        return reminders
```

Simple database operations (`faqbot`):
```python
async def save(self, evt: MessageEvent, key: str, value: str) -> None:
    if evt.sender not in self.config["adminlist"]:
        await evt.reply("You're not allowed to save FAQ entries.")
        return
    
    q = "INSERT INTO entries (roomid, key, value, global) VALUES ($1, $2, $3, 0)"
    await self.database.execute(q, evt.room_id, key, value)
    await evt.reply(f"Added FAQ entry for {key}")

async def _get_faq(self, evt: MessageEvent, key: str) -> None:
    q = "SELECT key, value FROM entries WHERE " \
        "LOWER(key)=LOWER($1) AND (roomid=$2 OR global=1)"
    row = await self.database.fetchrow(q, key, evt.room_id)
    if row:
        await evt.reply(f"{row['value']}")
    else:
        await evt.reply(f"No FAQ entry for `{key}` :(")
```

### Advanced Database Patterns from Real Projects

**1. In-Memory Caching with Database Persistence** (`reminder`):
```python
# Cache frequently accessed data to reduce database queries
class ReminderDatabase:
    cache: DefaultDict[UserID, UserInfo]
    
    async def get_user_info(self, user_id: UserID) -> UserInfo:
        if user_id not in self.cache:
            query = "SELECT timezone, locale FROM user_settings WHERE user_id = $1"
            row = dict(await self.db.fetchrow(query, user_id) or {})
            
            # Validate and use defaults for invalid data
            locale = row.get("locale", self.defaults.locale)
            if not locale or not validate_locale(locale):
                locale = self.defaults.locale
                
            self.cache[user_id] = UserInfo(locale=locale, timezone=timezone)
        return self.cache[user_id]
    
    async def set_user_info(self, user_id: UserID, key: str, value: str) -> None:
        # Update cache first
        await self.get_user_info(user_id)  # Ensure cached
        setattr(self.cache[user_id], key, value)
        
        # Then update database with UPSERT pattern
        q = """INSERT INTO user_settings (user_id, {0}) 
               VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET {0} = EXCLUDED.{0}"""
        await self.db.execute(q.format(key), user_id, value)
```

**2. Complex Data Loading with Relationships** (`reminder`):
```python
# Load related data efficiently with JOINs
async def load_all(self, bot: ReminderBot) -> Dict[EventID, Reminder]:
    rows = await self.db.fetch("""
        SELECT event_id, room_id, message, reply_to, start_time, recur_every, 
               cron_tab, is_agenda, user_id, confirmation_event, subscribing_event, creator
        FROM reminder NATURAL JOIN reminder_target
    """)
    
    reminders = {}
    for row in rows:
        # Handle one-to-many relationships in result processing
        if row["event_id"] in reminders:
            # Add subscriber to existing reminder
            reminders[row["event_id"]].subscribed_users[row["user_id"]] = row["subscribing_event"]
            continue
        
        # Data validation and cleanup during loading
        start_time = datetime.fromisoformat(row["start_time"]) if row["start_time"] else None
        if start_time and not row["is_agenda"] and not row["recur_every"] and not row["cron_tab"]:
            # Clean up expired one-time reminders
            if start_time < datetime.now(tz=pytz.UTC):
                await self.delete_reminder(row["event_id"])
                continue
        
        # Create object with loaded data
        reminders[row["event_id"]] = Reminder(...)
    return reminders
```

**3. Database Migration Patterns** (observed across projects):
```python
# Simple migration pattern
@upgrade_table.register(description="Initial revision")
async def upgrade_v1(conn: Connection) -> None:
    await conn.execute("""CREATE TABLE entries (
        roomid TEXT,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        UNIQUE(roomid, key) ON CONFLICT REPLACE
    )""")

# Schema evolution pattern
@upgrade_table.register(description="Add global entries support")
async def upgrade_v2(conn: Connection) -> None:
    await conn.execute("ALTER TABLE entries ADD COLUMN global INTEGER DEFAULT(0)")
    await conn.execute("CREATE INDEX global_entries ON entries(global)")
    await conn.execute("CREATE UNIQUE INDEX room_entries ON entries(roomid, key)")

# Complex migration with data transformation
@upgrade_table.register(description="Latest revision", upgrades_to=1)
async def upgrade_latest(conn: Connection, scheme: Scheme) -> None:
    needs_migration = False
    if await conn.table_exists("webhook"):
        needs_migration = True
        await conn.execute("ALTER TABLE webhook RENAME TO webhook_old;")
        # Create new schema...
    
    # Create new tables with improved schema
    await conn.execute("""CREATE TABLE webhook (
        id uuid NOT NULL,
        repo TEXT NOT NULL,
        user_id TEXT NOT NULL,
        room_id TEXT NOT NULL,
        secret TEXT NOT NULL,
        github_id INTEGER,
        PRIMARY KEY (id),
        CONSTRAINT webhook_repo_room_unique UNIQUE (repo, room_id)
    )""")
    
    if needs_migration:
        await migrate_legacy_to_v1(conn)
```

**4. UPSERT Patterns for Conflict Resolution** (`github`, `matrix-url-download`):
```python
# PostgreSQL UPSERT pattern
async def put_client(self, user_id: UserID, token: str) -> None:
    await self.db.execute("""
        INSERT INTO client (user_id, token) VALUES ($1, $2)
        ON CONFLICT (user_id) DO UPDATE SET token = excluded.token
    """, user_id, token)

# Conditional insert pattern
async def put_avatar(self, url: str, mxc: ContentURI) -> None:
    await self.db.execute("""
        INSERT INTO avatar (url, mxc) VALUES ($1, $2)
        ON CONFLICT (url) DO NOTHING
    """, url, mxc)
```

**5. Room State Management** (`faqbot`, `matrix-url-download`):
```python
# Room-based data with tombstone handling
@event.on(EventType.ROOM_TOMBSTONE)
async def tombstone(self, evt: StateEvent) -> None:
    if not evt.content.replacement_room:
        return
    
    # Migrate data to new room
    await self.database.execute(
        "UPDATE entries SET roomid=$1 WHERE roomid=$2",
        evt.content.replacement_room, evt.room_id
    )
    
    await self.client.send_notice(evt.room_id,
        f"Upgraded FAQ entries to room {evt.content.replacement_room}")
    await self.client.leave_room(evt.room_id)

# Room status tracking
async def ensure_in_room(self, room_id: RoomID) -> None:
    if not await self.is_in_room(room_id):
        await self.join_room(room_id)

async def is_in_room(self, room_id: RoomID) -> bool:
    rows = await self.db.fetch("SELECT enabled FROM status WHERE room_id = $1", room_id)
    return rows is not None and len(rows) > 0
```

**6. Database Error Handling and Resilience** (`maubot-communitybot`):
```python
# Retry logic for database operations
async def do_sync(self) -> None:
    try:
        space_members_obj = await self.client.get_joined_members(self.config["parent_room"])
        space_members_list = space_members_obj.keys()
    except asyncpg.exceptions.UniqueViolationError as e:
        # Handle specific database errors with retry
        self.log.warning(f"Duplicate key error during member sync, retrying: {e}")
        await asyncio.sleep(1)
        space_members_obj = await self.client.get_joined_members(self.config["parent_room"])
        space_members_list = space_members_obj.keys()
    except Exception as e:
        self.log.error(f"Failed to get space members: {e}")
        return {"added": [], "dropped": []}

# Batch operations with error handling
async def redact_messages(self, room_id):
    counters = {"success": 0, "failure": 0}
    events = await self.database.fetch(
        "SELECT event_id FROM redaction_tasks WHERE room_id = $1", room_id
    )
    
    for event in events:
        try:
            await self.client.redact(room_id, event["event_id"], reason="content removed")
            counters["success"] += 1
            await self.database.execute(
                "DELETE FROM redaction_tasks WHERE event_id = $1", event["event_id"]
            )
        except Exception as e:
            if "Too Many Requests" in str(e):
                self.log.warning(f"Rate limited, will try again in next loop")
                return counters
            self.log.error(f"Failed to redact message: {e}")
            counters["failure"] += 1
    return counters
```

**7. Transaction Usage for Complex Operations** (`github`):
```python
# Use transactions for multi-step operations
async def start(self) -> None:
    if await self.database.table_exists("needs_post_migration"):
        self.log.info("Running database post-migration")
        async with self.database.acquire() as conn, conn.transaction():
            await self.db.run_post_migration(conn, self.config["webhook_key"])
        self.log.info("Webhook secret migration completed successfully")

# Pass connection for transaction consistency
async def insert_webhook(self, webhook: WebhookInfo, *, _conn: Connection | None = None) -> None:
    await (_conn or self.db).execute("""
        INSERT INTO webhook (id, repo, user_id, room_id, secret, github_id)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, str(webhook.id), webhook.repo, webhook.user_id, 
        webhook.room_id, webhook.secret, webhook.github_id)
```

**8. Data Validation and Sanitization** (`reminder`):
```python
# Validate data before database storage
async def store_reminder(self, reminder: Reminder) -> None:
    await self.db.execute("""
        INSERT INTO reminder (event_id, room_id, start_time, message, reply_to, 
                             cron_tab, recur_every, is_agenda, creator)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """,
        reminder.event_id,
        reminder.room_id,
        # Sanitize datetime to remove microseconds for consistency
        reminder.start_time.replace(microsecond=0).isoformat() if reminder.start_time else None,
        reminder.message,
        reminder.reply_to,
        reminder.cron_tab,
        reminder.recur_every,
        reminder.is_agenda,
        reminder.creator
    )
```

**9. Database Performance Patterns**:
```python
# Use appropriate indexes for query patterns
@upgrade_table.register(description="Add performance indexes")
async def upgrade_v3(conn: Connection) -> None:
    # Index for frequent lookups
    await conn.execute("CREATE INDEX global_entries ON entries(global)")
    # Composite index for complex queries
    await conn.execute("CREATE UNIQUE INDEX room_entries ON entries(roomid, key)")
    # Index for foreign key relationships
    await conn.execute("CREATE INDEX reminder_target_event_id ON reminder_target(event_id)")

# Use fetchval for single value queries
async def get_event(self, message_id: str, room_id: RoomID) -> EventID | None:
    return await self.db.fetchval(
        "SELECT event_id FROM matrix_message WHERE message_id = $1 AND room_id = $2",
        message_id, room_id
    )

# Use fetch for multiple rows, fetchrow for single row
async def get_webhooks_in_room(self, room_id: RoomID) -> list[WebhookInfo]:
    rows = await self.db.fetch(
        "SELECT id, repo, user_id, room_id, github_id, secret FROM webhook WHERE room_id = $1",
        room_id
    )
    return [WebhookInfo.from_row(row) for row in rows]
```

### 🔧 Standard Operating Procedure

1.  **Enable database access in `maubot.yaml`** (`database: true`, `database_type: asyncpg`).
2.  **Create `UpgradeTable` and migration functions** using `@upgrade_table.register()`.
3.  **Implement `@classmethod get_db_upgrade_table()`** to return your upgrade table.
4.  **Use appropriate query methods**: `fetchval()` for single values, `fetchrow()` for single rows, `fetch()` for multiple rows.
5.  **Use `self.database.execute()`** for inserts, updates, and deletions.
6.  **Always use parameterized queries (`$1`, `$2`, etc.)** to prevent SQL injection.
7.  **Create dedicated database classes** for complex plugins to separate concerns.
8.  **Use `async with self.database.transaction():`** for atomic operations when needed.
9.  **Implement caching strategies** for frequently accessed data with cache invalidation.
10. **Use UPSERT patterns** (`ON CONFLICT ... DO UPDATE/NOTHING`) for conflict resolution.
11. **Handle database errors specifically** with retry logic for transient failures.
12. **Validate and sanitize data** before storage, especially datetime objects.
13. **Create appropriate indexes** for query performance in migrations.
14. **Handle room tombstones** by migrating data to replacement rooms.
15. **Use batch operations** with error handling for bulk data processing.

-----

## 6\. Performance Optimization and Security Hardening

### ❌ What Didn't Work

**Problem**: Recompiling regex patterns on every function call.

  - Performance bottleneck when processing many messages or URLs.
  - Unnecessary CPU overhead from repeated pattern compilation.
  - Example: URL validation regex compiled on every `process_url()` call.

**Problem**: Resource leaks in HTTP clients and database connections.

  - `aiohttp.ClientSession` instances created without proper timeout handling.
  - Database connections not properly managed in long-running operations.
  - Memory leaks from growing message ID maps without cleanup.

**Problem**: Race conditions in concurrent message processing.

  - Multiple messages processed simultaneously without locking mechanisms.
  - Shared state corruption when handling rapid message sequences.
  - No cleanup of processing locks leading to memory growth.

**Problem**: Missing input validation and rate limiting.

  - No sanitization of URLs or user input before processing.
  - No rate limiting on expensive operations like AI API calls.
  - Security vulnerabilities from unvalidated external data.

### ✅ What Worked

**Solution**: Compile patterns during initialization and implement proper resource management.

**1. Pattern Compilation Optimization**:
```python
# ❌ Wrong - compiles regex on every call
class Config(BaseProxyConfig):
    def is_valid_url(self, url: str) -> bool:
        pattern = re.compile(r'^https?://.+')  # Compiled every time!
        return bool(pattern.match(url))

# ✅ Correct - compile once during initialization
class Config(BaseProxyConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._url_pattern = None
        self._discourse_pattern = None
    
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        helper.copy("discourse_url")
        # Compile patterns after config update
        self._url_pattern = re.compile(r'^https?://.+')
        self._discourse_pattern = re.compile(rf'^{re.escape(self["discourse_url"])}/t/.+')
    
    def is_valid_url(self, url: str) -> bool:
        return bool(self._url_pattern.match(url))
```

**2. Proper HTTP Client Management**:
```python
# ❌ Wrong - creates new session for every request, causes resource leaks
async def make_api_call(self, data: dict) -> dict:
    async with aiohttp.ClientSession() as session:  # New session every time!
        async with session.post(url, json=data) as response:
            return await response.json()

# ❌ Wrong - no timeout handling, potential hanging requests
async def call_api(self, url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:  # No timeout!
            return await response.json()

# ✅ Correct - use plugin's HTTP client with proper timeout
class AIIntegration:
    def __init__(self, config: Config, http_client=None):
        self.config = config
        self.http = http_client  # Use plugin's HTTP client
    
    async def call_openai_api(self, messages: list) -> str:
        headers = {"Authorization": f"Bearer {self.config['openai_api_key']}"}
        data = {"model": self.config['openai_model'], "messages": messages}
        
        # Use plugin's HTTP client with proper timeout
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            if self.http:
                async with self.http.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers, json=data, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content']
            else:
                # Fallback only if necessary
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                        response.raise_for_status()
                        result = await response.json()
                        return result['choices'][0]['message']['content']
        except asyncio.TimeoutError:
            raise Exception("OpenAI API request timed out")
        except aiohttp.ClientResponseError as e:
            raise Exception(f"OpenAI API error: {e.status}")

# ✅ Correct - pass HTTP client to helper classes
class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        # Pass self.http to all helper classes
        self.ai_integration = AIIntegration(self.config, self.http)
        self.discourse_api = DiscourseAPI(self.config, self.http)
```

**Critical HTTP Session Management Rules**:
- **Never create `aiohttp.ClientSession()` in frequently called methods** - causes "Session is closed" errors
- **Always pass the plugin's `self.http` client to helper classes** during initialization
- **Use proper timeout handling** with `aiohttp.ClientTimeout(total=seconds)`
- **Provide fallbacks** for when HTTP client is not available, but prefer the plugin's client
- **Session closure errors** indicate improper session management - fix by using plugin's HTTP client

**3. Race Condition Prevention**:
```python
# ❌ Wrong - no locking, potential race conditions
class MatrixToDiscourseBot(Plugin):
    def __init__(self):
        self.processing_messages = set()
    
    async def handle_message(self, evt: MessageEvent):
        if evt.event_id in self.processing_messages:
            return
        self.processing_messages.add(evt.event_id)
        # Process message...
        self.processing_messages.remove(evt.event_id)

# ✅ Correct - proper locking with cleanup
class MatrixToDiscourseBot(Plugin):
    def __init__(self):
        self.message_locks = {}
        self.lock_cleanup_task = None
    
    async def start(self) -> None:
        await super().start()
        # Start periodic lock cleanup
        self.lock_cleanup_task = background_task.create(self._cleanup_locks_periodically())
    
    async def stop(self) -> None:
        if self.lock_cleanup_task:
            self.lock_cleanup_task.cancel()
        await super().stop()
    
    async def handle_message(self, evt: MessageEvent):
        # Create or get existing lock for this message
        if evt.event_id not in self.message_locks:
            self.message_locks[evt.event_id] = asyncio.Lock()
        
        async with self.message_locks[evt.event_id]:
            # Process message safely
            await self._process_message_content(evt)
    
    async def _cleanup_locks_periodically(self):
        while True:
            await asyncio.sleep(3600)  # Cleanup every hour
            # Remove old locks to prevent memory leaks
            current_time = time.time()
            old_locks = [
                event_id for event_id, lock in self.message_locks.items()
                if not lock.locked() and current_time - lock._created_time > 3600
            ]
            for event_id in old_locks:
                del self.message_locks[event_id]
```

**4. Input Validation and Rate Limiting**:
```python
# ✅ Comprehensive input validation
class Config(BaseProxyConfig):
    def validate_url(self, url: str) -> bool:
        if not url or len(url) > 2048:  # Basic length check
            return False
        if not self._url_pattern.match(url):
            return False
        # Additional security checks
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            return False  # Prevent SSRF
        return True

# ✅ Rate limiting for expensive operations
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def check_rate_limit(self, key: str) -> bool:
        now = time.time()
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < self.window_seconds
        ]
        
        if len(self.requests[key]) >= self.max_requests:
            return False  # Rate limited
        
        self.requests[key].append(now)
        return True
```

**5. Memory Leak Prevention**:
```python
# ✅ Proper cleanup of growing data structures
class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        await super().start()
        self.message_id_map = {}
        # Start periodic cleanup
        self.cleanup_task = background_task.create(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(1800)  # Cleanup every 30 minutes
            try:
                # Clean up old message mappings (older than 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_messages = [
                    msg_id for msg_id, timestamp in self.message_id_map.items()
                    if timestamp < cutoff_time
                ]
                for msg_id in old_messages:
                    del self.message_id_map[msg_id]
                
                self.log.debug(f"Cleaned up {len(old_messages)} old message mappings")
            except Exception as e:
                self.log.error(f"Error during periodic cleanup: {e}")
```

### 🔧 Standard Operating Procedure

1.  **Compile regex patterns during initialization**, not on every use.
2.  **Use plugin's HTTP client (`self.http`)** with proper timeout handling.
3.  **Implement locking mechanisms** for concurrent operations on shared state.
4.  **Add periodic cleanup tasks** to prevent memory leaks from growing data structures.
5.  **Validate all external input** including URLs, user data, and API responses.
6.  **Implement rate limiting** for expensive operations (AI APIs, external services).
7.  **Use proper exception handling** with specific timeout and error types.
8.  **Monitor resource usage** and implement cleanup strategies.
9.  **Test under load** to identify performance bottlenecks and race conditions.
10. **Use background tasks for cleanup** with proper cancellation in `stop()`.

-----

## 7\. Error Handling and Logging

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

## 8\. Testing and Deployment

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

### Plugin Building and Import Issues

**Problem**: Plugin fails to load with `main_class` not found errors.

  - Error: "Main class MatrixToDiscourseBot not in rate_limiter" - occurs when `main_class` doesn't specify module path and maubot looks in the wrong module.
  - Maubot looks for the main class in the **last** module listed in the `modules` array when no module prefix is specified.

**Problem**: Relative imports fail in maubot plugins.

  - Error: "ImportError: attempted relative import with no known parent package"
  - Maubot loads modules individually, not as a package, so relative imports (`from .module import ...`) don't work.

**Problem**: Module loading order dependency issues.

  - Error: "NameError: name 'upgrade_table' is not defined" - occurs when modules try to import from modules that haven't been loaded yet.
  - Maubot loads modules in the order they appear in the `modules` array, so dependencies must be listed first.

**Solution**: Proper `main_class` specification, absolute imports, and correct module ordering.

**`main_class` specification**:
```yaml
# ❌ Wrong - will look in last module (rate_limiter)
modules:
  - MatrixToDiscourseBot
  - db
  - rate_limiter
main_class: MatrixToDiscourseBot

# ✅ Correct - explicitly specify module
modules:
  - MatrixToDiscourseBot
  - db
  - rate_limiter
main_class: MatrixToDiscourseBot/MatrixToDiscourseBot
```

**Import patterns**:
```python
# ❌ Wrong - relative imports don't work
from .db import upgrade_table, MessageMappingDatabase
from .rate_limiter import RateLimiter

# ✅ Correct - use absolute imports
from db import upgrade_table, MessageMappingDatabase
from rate_limiter import RateLimiter
```

**Module loading order**:
```yaml
# ❌ Wrong - MatrixToDiscourseBot tries to import db before db is loaded
modules:
  - MatrixToDiscourseBot  # This imports from db
  - db                    # But db is loaded after

# ✅ Correct - Dependencies loaded first
modules:
  - db                    # Load db first
  - rate_limiter         # Load rate_limiter second
  - MatrixToDiscourseBot  # Load main module last (imports from db)
```

**Problem**: Configuration validation logic doesn't match actual config structure.

  - Error: "Missing required fields for local: api_endpoint, model" - occurs when validation logic uses wrong config key format.
  - Validation code looks for `local.api_endpoint` but config uses `local_llm.api_endpoint`.
  - Inconsistent naming between config sections and validation logic.

**Solution**: Ensure validation logic matches actual config structure.

```python
# ❌ Wrong - validation doesn't match config structure
def _validate_ai_config(self, model_type: str) -> bool:
    required_fields = {
        "local": ["api_endpoint", "model"]  # Looks for local.api_endpoint
    }
    config_key = f"{model_type}.{field}"  # Creates "local.api_endpoint"

# ✅ Correct - handle different config key formats
def _validate_ai_config(self, model_type: str) -> bool:
    required_fields = {
        "local": ["api_endpoint", "model"]
    }
    # Handle the different config key format for local_llm
    if model_type == "local":
        config_key = f"local_llm.{field}"  # Creates "local_llm.api_endpoint"
    else:
        config_key = f"{model_type}.{field}"
```

**Building plugins without mbc**:
If you don't have `mbc` installed, you can create build scripts:

```python
#!/usr/bin/env python3
"""Build script for maubot plugin without requiring mbc."""
import zipfile
import yaml
from pathlib import Path

def build_plugin():
    # Load metadata
    with open("plugin/maubot.yaml", "r") as f:
        meta = yaml.safe_load(f)
    
    plugin_id = meta.get("id", "unknown")
    version = meta.get("version", "0.0.0")
    output_file = f"{plugin_id}-v{version}.mbp"
    
    # Create ZIP file
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as mbp:
        # Add maubot.yaml
        mbp.write("plugin/maubot.yaml", "maubot.yaml")
        
        # Add modules
        for module in meta.get("modules", []):
            if Path(f"plugin/{module}.py").exists():
                mbp.write(f"plugin/{module}.py", f"{module}.py")
        
        # Add config if exists
        if meta.get("config") and Path("plugin/base-config.yaml").exists():
            mbp.write("plugin/base-config.yaml", "base-config.yaml")
    
    print(f"Built {output_file}")

if __name__ == "__main__":
    build_plugin()
```

**Critical: Always rebuild after fixing imports**:
When you fix import issues, you MUST rebuild the .mbp file. The error messages will continue to show the old import errors until you rebuild and re-upload the plugin, because the .mbp file contains the old version of your code.

**Problem**: Multiple event handlers for the same event type causing conflicts.

  - Having multiple `@event.on(EventType.ROOM_MESSAGE)` decorators can cause unpredictable behavior.
  - Event handlers may not trigger as expected or may interfere with each other.
  - URL processing may fail silently without any logs or error messages.
  - Only one handler may execute, or handlers may execute in unexpected order.

**Solution**: Use a single event handler and route internally.

```python
# ❌ Wrong - Multiple event handlers for same event type
@event.on(EventType.ROOM_MESSAGE)
async def handle_message(self, evt: MessageEvent) -> None:
    # URL processing logic
    if urls := extract_urls(evt.content.body):
        await self.process_urls(evt, urls)

@event.on(EventType.ROOM_MESSAGE)  # Conflict! May not execute
async def handle_matrix_reply(self, evt: MessageEvent) -> None:
    # Reply processing logic
    if evt.content.get_reply_to():
        await self.process_reply(evt)

# ✅ Correct - Single event handler with internal routing
@event.on(EventType.ROOM_MESSAGE)
async def handle_message(self, evt: MessageEvent) -> None:
    # Process URLs if found
    if urls := extract_urls(evt.content.body):
        await self.process_urls(evt, urls)
    
    # Also check for replies (both can happen in same message)
    await self.handle_matrix_reply(evt)

async def handle_matrix_reply(self, evt: MessageEvent) -> None:
    """Handle replies - called from main handler, not as separate event handler."""
    if not evt.content.get_reply_to():
        return  # Not a reply
    await self.process_reply(evt)
```

**Module loading order matters**:
In `maubot.yaml`, list modules in dependency order. If `ModuleA` imports from `ModuleB`, then `ModuleB` must be listed before `ModuleA` in the `modules` array. Maubot loads modules in the order they're listed, so dependencies must be loaded first.

```yaml
# ❌ Wrong - MatrixToDiscourseBot tries to import db before db is loaded
modules:
  - MatrixToDiscourseBot  # This imports from db
  - db                    # But db is loaded after

# ✅ Correct - Dependencies loaded first
modules:
  - db                    # Load db first
  - rate_limiter         # Load rate_limiter second
  - MatrixToDiscourseBot  # Load main module last (imports from db)
```

### 🔧 Standard Operating Procedure

1.  **Develop in a Python virtual environment.**
2.  **Write unit tests** for your plugin's core logic, mocking Maubot dependencies. Use `pytest` and `pytest-asyncio`.
3.  **Use `mbc build` to generate `.mbp` files** for release.
4.  **During active development, use `mbc build --upload`** for quick iterations.
5.  **Test plugins in a dedicated test Matrix room** with a separate bot account/instance to avoid affecting production.
6.  **Familiarize yourself with Maubot's web management UI** for uploading, creating instances, and viewing logs in a production environment.
7.  **Always specify the full module path in `main_class`** when using multiple modules (e.g., `ModuleName/ClassName`).
8.  **Use absolute imports only** - avoid relative imports (`from .module import ...`) as they don't work in maubot.
9.  **Order modules by dependency** - list dependencies first in the `modules` array.
10. **Ensure validation logic matches config structure** - check that validation code uses correct config key formats.
11. **Always rebuild after fixing imports** - the .mbp file contains cached code until rebuilt.
12. **Use single event handlers** - avoid multiple `@event.on()` decorators for the same event type.
13. **Add debug logging for troubleshooting** - use `logger.info()` for important events and `logger.debug()` for detailed tracing.
14. **Create build scripts** as backup if `mbc` is not available or for CI/CD automation.

-----

## 9\. Plugin Architecture Patterns

### ❌ What Didn't Work

**Problem**: Inconsistent file organization and architecture patterns.

  - Mixing simple and complex patterns inappropriately.
  - Not following established conventions from successful plugins.
  - Reinventing patterns that already exist in the ecosystem.

### ✅ What Worked

**Solution**: Follow established architecture patterns based on plugin complexity.

**Simple Single-File Plugins** (like `timer`, `faqbot`):
```
plugin_name.py
maubot.yaml
base-config.yaml (optional)
LICENSE
README.md
```

**Medium Complexity Plugins** (like `reactbot`):
```
plugin_name/
├── __init__.py
├── bot.py
├── config.py
├── rule.py
├── template.py
maubot.yaml
base-config.yaml
pyproject.toml
.pre-commit-config.yaml
```

**Complex Multi-Module Plugins** (like `reminder`, `github`):
```
plugin_name/
├── __init__.py
├── bot.py (main plugin class)
├── config.py
├── db.py
├── migrations.py
├── util.py
├── api/ (for external API integrations)
├── webhook/ (for webhook handling)
├── template/ (for message templates)
maubot.yaml
base-config.yaml
pyproject.toml
.pre-commit-config.yaml
.gitlab-ci.yml
```

### Common Patterns Observed

**1. Standardized CI/CD**:
Most modern projects use GitHub Actions instead of GitLab CI. For GitHub-based projects:
```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install black isort flake8
    - run: black --check plugin/
    - run: isort --check-only plugin/
```

Note: Some older maubot plugins may still use GitLab CI with:
```yaml
include:
- project: 'maubot/maubot'
  file: '/.gitlab-ci-plugin.yml'
```

**2. Code Quality Tools**:
Standard `pyproject.toml` configuration:
```toml
[tool.isort]
profile = "black"
force_to_top = "typing"
from_first = true
combine_as_imports = true
known_first_party = ["mautrix", "maubot"]
line_length = 99

[tool.black]
line-length = 99
target-version = ["py310"]
```

**3. Pre-commit Hooks**:
```yaml
repos:
- repo: https://github.com/psf/black
  rev: 22.12.0
  hooks:
  - id: black
- repo: https://github.com/pycqa/isort
  rev: 5.11.4
  hooks:
  - id: isort
```

**4. Licensing**:
Consistent use of `AGPL-3.0-or-later` across most plugins.

**5. Import Patterns**:
Standard imports observed across plugins:
```python
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event, web
from mautrix.types import EventType, TextMessageEventContent, UserID, RoomID
from mautrix.util.async_db import UpgradeTable, Connection
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper
from mautrix.util import background_task
```

**6. Component Separation**:
- `bot.py`: Main plugin class and command handlers
- `config.py`: Configuration management
- `db.py`: Database operations and models
- `migrations.py`: Database schema migrations
- `util.py`: Helper functions and utilities

### 🔧 Standard Operating Procedure

1.  **Choose architecture based on complexity**: Single-file for simple plugins, multi-module for complex ones.
2.  **Follow established file naming conventions**: `bot.py`, `config.py`, `db.py`, `migrations.py`.
3.  **Use appropriate CI/CD for your platform**: GitHub Actions for GitHub repos, GitLab CI for GitLab repos.
4.  **Implement code quality tools**: Use `pyproject.toml` with black and isort configuration.
5.  **Use consistent licensing**: Prefer `AGPL-3.0-or-later` for compatibility.
6.  **Separate concerns properly**: Keep database, config, and business logic in separate modules.
7.  **Follow import conventions**: Use the standard maubot/mautrix import patterns.

-----

## 10\. Multi-File Plugin Organization

### ❌ What Didn't Work

**Problem**: Putting everything in a single file for complex plugins.

  - Files become unwieldy (1000+ lines) and difficult to navigate.
  - Mixing concerns (database, API clients, webhooks, commands) in one file.
  - Difficult to test individual components in isolation.
  - Hard to maintain and refactor as complexity grows.

**Problem**: Inconsistent file organization across plugins.

  - No clear patterns for where to put different types of functionality.
  - Difficulty onboarding new developers to the codebase.
  - Reinventing organizational patterns for each plugin.

### ✅ What Worked

**Solution**: Follow established multi-file organization patterns based on plugin complexity.

**File Organization Patterns Observed**:

**Simple Plugins (< 200 lines)**:
```
plugin_name.py          # Single file with all functionality
maubot.yaml
base-config.yaml (optional)
```

**Medium Plugins (200-800 lines)**:
```
plugin_name/
├── __init__.py         # from .bot import PluginBot
├── bot.py              # Main plugin class and command handlers
├── config.py           # Configuration management
├── db.py               # Database operations (optional)
maubot.yaml
base-config.yaml
```

**Complex Plugins (800+ lines)**:
```
plugin_name/
├── __init__.py         # from .bot import PluginBot
├── bot.py              # Main plugin class and lifecycle
├── config.py           # Configuration management
├── db.py               # Database operations and models
├── migrations.py       # Database schema migrations
├── commands.py         # Command handlers
├── client_manager.py   # External service clients
├── api/                # External API integration
│   ├── __init__.py     # from .client import APIClient
│   ├── client.py       # HTTP client for external APIs
│   ├── types.py        # Data models and types
│   └── webhook.py      # Webhook receiver logic
├── webhook/            # Webhook handling (if complex)
│   ├── __init__.py     # from .handler import WebhookHandler
│   ├── handler.py      # Main webhook processing
│   ├── manager.py      # Webhook management
│   └── aggregation.py  # Event aggregation logic
├── template/           # Message templates
│   ├── __init__.py     # from .manager import TemplateManager
│   ├── manager.py      # Template management
│   └── util.py         # Template utilities
└── util/               # Shared utilities
    ├── __init__.py     # Utility exports
    ├── contrast.py     # Color utilities
    └── recursive_get.py # Data access helpers
maubot.yaml
base-config.yaml
```

**Real-world examples from analyzed plugins**:

GitHub plugin structure (complex):
```python
# github/github/__init__.py
from .bot import GitHubBot

# github/github/bot.py
class GitHubBot(Plugin):
    def __init__(self):
        self.db = DBManager(self.database)
        self.clients = ClientManager(...)
        self.webhook_manager = WebhookManager(self.db)
        self.webhook_handler = WebhookHandler(bot=self)
        self.avatars = AvatarManager(bot=self)
        self.webhook_receiver = GitHubWebhookReceiver(...)
        self.commands = Commands(bot=self)

# github/github/api/__init__.py
from .client import GitHubClient, GitHubError, GraphQLError
from .webhook import GitHubWebhookReceiver

# github/github/webhook/__init__.py
from .aggregation import PendingAggregation
from .handler import WebhookHandler
from .manager import WebhookInfo, WebhookManager
```

Reminder plugin structure (medium):
```python
# reminder/reminder/__init__.py
from .bot import ReminderBot

# reminder/reminder/bot.py
class ReminderBot(Plugin):
    def __init__(self):
        self.db = ReminderDatabase(self.database)
        self.scheduler = AsyncIOScheduler()
```

### Component Separation Patterns

**1. Main Plugin Class (`bot.py`)**:
- Plugin lifecycle (`start()`, `stop()`)
- Component initialization and coordination
- High-level command routing
- Configuration management integration

**2. Command Handlers (`commands.py` or in `bot.py`)**:
- All `@command.new()` decorated methods
- Command argument parsing and validation
- User permission checks
- Delegation to business logic components

**3. Database Layer (`db.py`)**:
- Database models and operations
- Migration definitions
- Data access patterns
- Query optimization

**4. External Service Integration (`api/` directory)**:
- HTTP clients for external APIs
- Authentication and token management
- Rate limiting and retry logic
- Data model serialization/deserialization

**5. Webhook Handling (`webhook/` directory)**:
- Webhook signature verification
- Event parsing and validation
- Event aggregation and deduplication
- Response formatting

**6. Configuration Management (`config.py`)**:
- `BaseProxyConfig` subclass
- Configuration validation
- Default value management
- Runtime configuration updates

### Module Import Patterns

**Standard `__init__.py` patterns observed**:
```python
# Simple export pattern
from .bot import PluginBot

# Multiple exports pattern
from .client import APIClient, APIError
from .webhook import WebhookReceiver

# Namespace organization pattern
from .util import contrast, hex_to_rgb, rgb_to_hex
from .util import recursive_get
```

**Cross-module dependency patterns**:
```python
# In bot.py - dependency injection pattern
class GitHubBot(Plugin):
    async def start(self) -> None:
        self.db = DBManager(self.database)
        self.clients = ClientManager(config, self.http, self.db)
        self.webhook_handler = WebhookHandler(bot=self)
        self.commands = Commands(bot=self)

# In commands.py - receive dependencies
class Commands:
    def __init__(self, bot: GitHubBot):
        self.bot = bot
        self.db = bot.db
        self.clients = bot.clients
```

### 🔧 Standard Operating Procedure

1.  **Start with single file** for prototypes and simple plugins (< 200 lines).
2.  **Split into medium structure** when approaching 200-300 lines or adding database.
3.  **Use complex structure** for plugins with external APIs, webhooks, or 800+ lines.
4.  **Follow naming conventions**: `bot.py`, `config.py`, `db.py`, `commands.py`, `migrations.py`.
5.  **Use directory-based organization** for related functionality (`api/`, `webhook/`, `template/`).
6.  **Implement proper `__init__.py` exports** for clean imports.
7.  **Use dependency injection** to pass shared components between modules.
8.  **Keep main plugin class lightweight** - delegate to specialized components.

-----

## 11\. Service Integration and Inter-Service Communication

### ❌ What Didn't Work

**Problem**: Hardcoding external service URLs and authentication.

  - Difficult to test with different environments.
  - Security risks from exposed credentials.
  - Inflexible integration patterns.

**Problem**: Poor error handling for external service failures.

  - Plugins becoming unresponsive when external services are down.
  - No retry logic or graceful degradation.
  - Poor user experience during service outages.

**Problem**: Inefficient webhook handling and verification.

  - Security vulnerabilities from unverified webhooks.
  - Performance issues from synchronous webhook processing.
  - No deduplication or aggregation of related events.

### ✅ What Worked

**Solution**: Implement robust service integration patterns with proper authentication, error handling, and webhook management.

### External Service Integration Patterns

**1. HTTP Client Management**:
```python
# Use self.http for external requests with proper error handling
class TwitchBot(Plugin):
    async def twitch_api(self, method: str, endpoint: str, params: dict | None, 
                         data: dict | None, evt: MessageEvent | None) -> dict | None:
        # Get access token from database
        q = "SELECT value FROM twitchbot_data WHERE key = $1"
        row = await self.database.fetchrow(q, "access_token")
        if row:
            access_token = row["value"]
        else:
            access_token = await self.get_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Client-Id": self.config["client_id"]
        }

        retry = 2
        while retry > 0:
            async with self.http.request(method, f"https://api.twitch.tv/helix{endpoint}",
                                       headers=headers, params=params, json=data) as response:
                if response.status == 401:
                    # Token expired, refresh and retry
                    access_token = await self.get_access_token()
                    headers["Authorization"] = f"Bearer {access_token}"
                    retry -= 1
                    continue
                elif response.status >= 400:
                    self.log.error(f"{method} {response.url}: {response.status}")
                    return None
                return await response.json()
        return None
```

**2. Authentication and Token Management**:
```python
# Store tokens securely in database with refresh logic
class GitHubClient:
    async def get_access_token(self) -> str:
        data = {
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "grant_type": "client_credentials"
        }
        async with self.http.post("https://id.twitch.tv/oauth2/token", data=data) as response:
            result = await response.json()
            if response.status >= 400:
                self.log.error(f"Error getting access token: {response.status}")
                return ""
            
            # Store new token in database
            q = "INSERT INTO tokens (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value=excluded.value"
            await self.database.execute(q, "access_token", result["access_token"])
            return result["access_token"]

    async def reset_token(self) -> Optional[str]:
        resp = await self.http.patch(self._token_url, json={"access_token": self.token})
        resp_data = await resp.json()
        if resp.status == 404:
            return None
        self.token = resp_data["token"]
        return self.token
```

**3. Rate Limiting and Retry Logic**:
```python
# Implement exponential backoff and rate limiting
class APIClient:
    def __init__(self):
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self.last_request_time = 0
        self.min_interval = 0.1  # Minimum time between requests

    async def make_request(self, method: str, url: str, **kwargs):
        async with self.rate_limiter:
            # Ensure minimum interval between requests
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            for attempt in range(3):
                try:
                    async with self.http.request(method, url, **kwargs) as response:
                        if response.status == 429:  # Rate limited
                            retry_after = int(response.headers.get('Retry-After', 60))
                            await asyncio.sleep(retry_after)
                            continue
                        response.raise_for_status()
                        self.last_request_time = time.time()
                        return await response.json()
                except aiohttp.ClientError as e:
                    if attempt == 2:  # Last attempt
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Webhook Integration Patterns

**1. Webhook Signature Verification**:
```python
# Always verify webhook signatures for security
class GitHubWebhookReceiver:
    async def _handle(self, request: web.Request, webhook_info: WebhookInfo) -> web.Response:
        try:
            signature = request.headers["X-Hub-Signature"]
            delivery_id = request.headers["X-Github-Delivery"]
        except KeyError as e:
            return web.Response(status=400, text=f"Missing {e.args[0]} header")

        text = await request.text()
        text_binary = text.encode("utf-8")
        secret = webhook_info.secret.encode("utf-8")
        digest = f"sha1={hmac.new(secret, text_binary, hashlib.sha1).hexdigest()}"
        
        if not hmac.compare_digest(signature, digest):
            return web.Response(status=401, text="Invalid signature")
        
        # Process webhook...
        return web.Response(status=200)
```

**2. Webhook Event Aggregation**:
```python
# Aggregate related webhook events to reduce noise
class PendingAggregation:
    def __init__(self, timeout: int):
        self.events = []
        self.timeout = timeout
        self.timer_task = None

    async def add_event(self, event: Event) -> None:
        self.events.append(event)
        
        if self.timer_task:
            self.timer_task.cancel()
        
        self.timer_task = background_task.create(self._send_after_timeout())

    async def _send_after_timeout(self) -> None:
        await asyncio.sleep(self.timeout)
        if self.events:
            await self._send_aggregated_message()
            self.events.clear()
```

**3. Webhook URL Management**:
```python
# Dynamic webhook URL generation and management
class WebhookManager:
    async def create_webhook(self, owner: str, repo: str) -> Webhook:
        webhook_id = uuid.uuid4()
        secret = self._generate_secret()
        
        # Store webhook info in database
        await self.db.store_webhook(webhook_id, owner, repo, secret)
        
        # Create webhook on external service
        webhook_url = self.base_url / "webhook" / str(webhook_id)
        return await self.api_client.create_webhook(
            owner, repo, webhook_url, secret=secret
        )

    def _generate_secret(self) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=64))
```

### Service Discovery and Health Checks

**1. Service Health Monitoring**:
```python
# Monitor external service health
class ServiceHealthChecker:
    async def check_service_health(self) -> bool:
        try:
            async with self.http.get(f"{self.base_url}/health", timeout=5) as response:
                return response.status == 200
        except asyncio.TimeoutError:
            self.log.warning("Service health check timed out")
            return False
        except Exception as e:
            self.log.error(f"Service health check failed: {e}")
            return False

    async def start_health_monitoring(self) -> None:
        self.health_check_task = background_task.create(self._health_check_loop())

    async def _health_check_loop(self) -> None:
        while True:
            is_healthy = await self.check_service_health()
            if not is_healthy:
                await self._handle_service_down()
            await asyncio.sleep(300)  # Check every 5 minutes
```

### Inter-Plugin Communication

**1. Matrix Room-Based Communication**:
```python
# Use Matrix rooms for inter-plugin communication
class PluginCommunicator:
    async def send_to_plugin(self, target_plugin: str, message: dict) -> None:
        communication_room = self.config["plugin_communication_room"]
        content = TextMessageEventContent(
            msgtype=MessageType.NOTICE,
            body=json.dumps({
                "target": target_plugin,
                "source": self.plugin_id,
                "data": message
            })
        )
        await self.client.send_message(communication_room, content)

    @event.on(EventType.ROOM_MESSAGE)
    async def handle_plugin_message(self, evt: MessageEvent) -> None:
        if evt.room_id != self.config["plugin_communication_room"]:
            return
        
        try:
            data = json.loads(evt.content.body)
            if data.get("target") == self.plugin_id:
                await self._process_plugin_message(data)
        except (json.JSONDecodeError, KeyError):
            pass  # Ignore malformed messages
```

**2. Database-Based Communication**:
```python
# Use shared database tables for plugin coordination
class PluginCoordinator:
    async def register_plugin(self) -> None:
        await self.database.execute(
            "INSERT INTO plugin_registry (plugin_id, status, last_seen) VALUES ($1, $2, $3) "
            "ON CONFLICT (plugin_id) DO UPDATE SET status=excluded.status, last_seen=excluded.last_seen",
            self.plugin_id, "active", datetime.utcnow()
        )

    async def send_event(self, event_type: str, data: dict) -> None:
        await self.database.execute(
            "INSERT INTO plugin_events (event_type, source_plugin, data, created_at) VALUES ($1, $2, $3, $4)",
            event_type, self.plugin_id, json.dumps(data), datetime.utcnow()
        )

    async def poll_events(self) -> List[dict]:
        rows = await self.database.fetch(
            "SELECT * FROM plugin_events WHERE target_plugin = $1 OR target_plugin IS NULL ORDER BY created_at",
            self.plugin_id
        )
        return [dict(row) for row in rows]
```

### 🔧 Standard Operating Procedure

1.  **Use `self.http` for all external HTTP requests** with proper error handling.
2.  **Store authentication tokens in database** with refresh logic.
3.  **Implement retry logic with exponential backoff** for transient failures.
4.  **Always verify webhook signatures** using HMAC comparison.
5.  **Use event aggregation** to reduce notification noise.
6.  **Monitor external service health** with background tasks.
7.  **Implement graceful degradation** when external services are unavailable.
8.  **Use Matrix rooms or database** for inter-plugin communication.
9.  **Generate unique webhook URLs** with proper secret management.
10. **Log all external service interactions** for debugging and monitoring.

-----

## 12\. Matrix Message Handling, Encryption, and Security

### ❌ What Didn't Work

**Problem**: Ignoring Matrix encryption and security considerations.

  - Plugins failing in encrypted rooms.
  - Security vulnerabilities from improper message handling.
  - Poor user experience in mixed encrypted/unencrypted environments.

**Problem**: Inadequate message content handling.

  - Not handling different message types (text, files, reactions).
  - Ignoring message formatting (HTML, Markdown).
  - Poor handling of message edits and redactions.

**Problem**: Insufficient user verification and permission checks.

  - Allowing unauthorized users to execute admin commands.
  - No protection against spam or abuse.
  - Inadequate rate limiting.

### ✅ What Worked

**Solution**: Implement comprehensive Matrix message handling with proper encryption support, security measures, and user verification.

### Matrix Message Types and Content Handling

**1. Comprehensive Message Type Support**:
```python
# Handle different Matrix message types appropriately
@event.on(EventType.ROOM_MESSAGE)
async def handle_message(self, evt: MessageEvent) -> None:
    if evt.sender == self.client.mxid:
        return  # Ignore own messages

    # Handle different message types
    if evt.content.msgtype == MessageType.TEXT:
        await self._handle_text_message(evt)
    elif evt.content.msgtype == MessageType.IMAGE:
        await self._handle_image_message(evt)
    elif evt.content.msgtype == MessageType.FILE:
        await self._handle_file_message(evt)
    elif evt.content.msgtype == MessageType.VIDEO:
        await self._handle_video_message(evt)
    elif evt.content.msgtype == MessageType.AUDIO:
        await self._handle_audio_message(evt)

async def _handle_file_message(self, evt: MessageEvent) -> None:
    # Check if file uploads should be censored
    if self._should_censor_files(evt.room_id, evt.sender):
        await self.client.redact(evt.room_id, evt.event_id, reason="File upload censored")
        return
    
    # Process file if needed
    if evt.content.info and evt.content.info.size > self.config["max_file_size"]:
        await evt.reply("File too large for processing")
```

**2. Message Content Processing**:
```python
# Handle both plain text and formatted content
async def process_message_content(self, evt: MessageEvent) -> str:
    # Prefer formatted content if available
    if evt.content.formatted_body and evt.content.format == Format.HTML:
        # Strip HTML tags for processing but preserve for display
        import re
        plain_text = re.sub(r'<[^>]+>', '', evt.content.formatted_body)
        return plain_text
    else:
        return evt.content.body

# Send messages with proper formatting
async def send_formatted_message(self, room_id: RoomID, message: str, html: str = None) -> None:
    if html:
        content = TextMessageEventContent(
            msgtype=MessageType.NOTICE,
            body=message,
            format=Format.HTML,
            formatted_body=html
        )
    else:
        content = TextMessageEventContent(
            msgtype=MessageType.NOTICE,
            body=message
        )
    await self.client.send_message(room_id, content)
```

**3. Message Reactions and Edits**:
```python
# Handle message reactions for subscriptions/interactions
@command.passive(regex=r"(?:\U0001F44D[\U0001F3FB-\U0001F3FF]?)",
                 field=lambda evt: evt.content.relates_to.key,
                 event_type=EventType.REACTION, msgtypes=None)
async def handle_thumbs_up(self, evt: ReactionEvent, _: Tuple[str]) -> None:
    # Subscribe user to reminder when they react with thumbs up
    original_event = evt.content.relates_to.event_id
    reminder = await self.db.get_reminder_by_event_id(original_event)
    if reminder:
        await reminder.subscribe_user(evt.sender)
        await evt.respond("You've been subscribed to this reminder!")

# Handle message edits
@event.on(EventType.ROOM_MESSAGE)
async def handle_message_edit(self, evt: MessageEvent) -> None:
    if evt.content.relates_to and evt.content.relates_to.rel_type == RelationType.REPLACE:
        original_event_id = evt.content.relates_to.event_id
        # Update any stored references to the original message
        await self.db.update_message_reference(original_event_id, evt.event_id)
```

### Encryption Support

**1. Encryption-Aware Room Creation**:
```python
# Create rooms with proper encryption settings
async def create_room(self, roomname: str, encrypt: bool = None) -> RoomID:
    if encrypt is None:
        encrypt = self.config["encrypt"]
    
    initial_state = []
    
    # Add encryption state event if needed
    if encrypt:
        initial_state.append({
            "type": str(EventType.ROOM_ENCRYPTION),
            "content": {
                "algorithm": "m.megolm.v1.aes-sha2"
            }
        })
    
    room_id = await self.client.create_room(
        name=roomname,
        initial_state=initial_state
    )
    
    return room_id

# Handle encryption status in greetings
async def send_greeting(self, room_id: RoomID, user_id: UserID) -> None:
    # Check if room is encrypted
    try:
        encryption_event = await self.client.get_state_event(
            room_id, EventType.ROOM_ENCRYPTION
        )
        is_encrypted = encryption_event.algorithm == "m.megolm.v1.aes-sha2"
    except Exception:
        is_encrypted = False
    
    if is_encrypted:
        greeting = self.config["greetings"]["encrypted"]
    else:
        greeting = self.config["greetings"]["generic"]
    
    await self.send_formatted_message(room_id, greeting.format(user=user_id))
```

**2. Encryption-Compatible Message Handling**:
```python
# Handle messages in encrypted rooms properly
@event.on(EventType.ROOM_MESSAGE)
async def handle_encrypted_message(self, evt: MessageEvent) -> None:
    # In encrypted rooms, message content is automatically decrypted by mautrix
    # No special handling needed, but be aware of potential decryption failures
    
    if not evt.content.body:
        # Message might be encrypted and not yet decrypted
        self.log.debug(f"Received message with no body in {evt.room_id}")
        return
    
    # Process normally
    await self.process_message(evt)
```

### User Verification and Security

**1. Human Verification System**:
```python
# Implement anti-spam human verification
@event.on(InternalEventType.JOIN)
async def handle_new_join(self, evt: StateEvent) -> None:
    if not self._verification_enabled(evt.room_id):
        return
    
    # Check if user needs verification
    power_levels = await self.client.get_state_event(
        evt.room_id, EventType.ROOM_POWER_LEVELS
    )
    user_level = power_levels.get_user_level(evt.sender)
    required_level = power_levels.events.get(str(EventType.ROOM_MESSAGE), 
                                           power_levels.events_default)
    
    if user_level >= required_level:
        return  # User already has permission
    
    # Start verification process
    await self._start_verification(evt.sender, evt.room_id)

async def _start_verification(self, user_id: UserID, target_room: RoomID) -> None:
    # Create DM room for verification
    dm_room = await self.client.create_room(
        preset=RoomCreatePreset.PRIVATE,
        invitees=[user_id],
        is_direct=True
    )
    
    # Send verification challenge
    phrase = random.choice(self.config["verification_phrases"])
    message = self.config["verification_message"].format(
        room=target_room, phrase=phrase
    )
    
    await self.client.send_notice(dm_room, html=message)
    
    # Store verification state
    await self._store_verification_state(dm_room, user_id, target_room, phrase)
```

**2. Permission and Rate Limiting**:
```python
# Implement comprehensive permission checking
async def check_user_permission(self, evt: MessageEvent, required_level: str = "user") -> bool:
    if required_level == "admin":
        return evt.sender in self.config["adminlist"]
    elif required_level == "moderator":
        return (evt.sender in self.config["adminlist"] or 
                evt.sender in self.config["moderators"])
    elif required_level == "power":
        # Check Matrix power level
        try:
            power_levels = await self.client.get_state_event(
                evt.room_id, EventType.ROOM_POWER_LEVELS
            )
            user_level = power_levels.get_user_level(evt.sender)
            return user_level >= self.config["required_power_level"]
        except Exception:
            return False
    return True  # Default allow for "user" level

# Rate limiting implementation
class RateLimiter:
    def __init__(self, max_requests: int, window_minutes: int):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = {}  # user_id -> list of timestamps

    async def check_rate_limit(self, user_id: UserID) -> bool:
        now = time.time()
        user_requests = self.requests.get(user_id, [])
        
        # Remove old requests outside the window
        user_requests = [req_time for req_time in user_requests 
                        if now - req_time < self.window_seconds]
        
        if len(user_requests) >= self.max_requests:
            return False  # Rate limited
        
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True
```

**3. Content Moderation and Censorship**:
```python
# Implement content moderation
async def check_message_content(self, evt: MessageEvent) -> bool:
    if not self._censorship_enabled(evt.room_id):
        return True
    
    # Check user power level for censorship exemption
    power_levels = await self.client.get_state_event(
        evt.room_id, EventType.ROOM_POWER_LEVELS
    )
    user_level = power_levels.get_user_level(evt.sender)
    
    if user_level >= self.config["uncensor_pl"]:
        return True  # User exempt from censorship
    
    # Check against word list
    message_text = evt.content.body.lower()
    for word in self.config["censor_wordlist"]:
        if word.lower() in message_text:
            await self.client.redact(evt.room_id, evt.event_id, 
                                   reason="Message contained prohibited content")
            return False
    
    return True

# File upload censorship
async def handle_file_upload(self, evt: MessageEvent) -> None:
    if (self.config["censor_files"] and 
        self._censorship_enabled(evt.room_id) and
        not self._user_exempt_from_censorship(evt.sender, evt.room_id)):
        
        await self.client.redact(evt.room_id, evt.event_id, 
                               reason="File uploads not permitted")
```

### Message Signing and Verification

**1. Message Integrity Verification**:
```python
# Verify message integrity for critical operations
async def verify_command_integrity(self, evt: MessageEvent) -> bool:
    # For critical commands, verify the message hasn't been tampered with
    # This is mainly relevant for webhook payloads or external integrations
    
    if hasattr(evt.content, 'signature'):
        # Verify signature if present
        message_data = evt.content.body.encode('utf-8')
        expected_signature = hmac.new(
            self.config["signing_key"].encode('utf-8'),
            message_data,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(evt.content.signature, expected_signature)
    
    return True  # No signature required for regular messages

# Sign outgoing webhook payloads
async def send_webhook_payload(self, url: str, data: dict) -> None:
    payload = json.dumps(data).encode('utf-8')
    signature = hmac.new(
        self.config["webhook_secret"].encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        'Content-Type': 'application/json',
        'X-Hub-Signature-256': f'sha256={signature}'
    }
    
    async with self.http.post(url, data=payload, headers=headers) as response:
        if response.status != 200:
            self.log.error(f"Webhook delivery failed: {response.status}")
```

### 🔧 Standard Operating Procedure

1.  **Handle all Matrix message types** (text, image, file, video, audio).
2.  **Support both plain text and HTML formatted content** in messages.
3.  **Create encrypted rooms** using `m.megolm.v1.aes-sha2` algorithm when needed.
4.  **Implement user verification systems** for anti-spam protection.
5.  **Check user permissions** before executing privileged commands.
6.  **Implement rate limiting** to prevent abuse.
7.  **Use content moderation** with configurable word lists and file restrictions.
8.  **Handle message reactions and edits** appropriately.
9.  **Verify webhook signatures** using HMAC for security.
10. **Log security events** for monitoring and auditing.
11. **Gracefully handle encryption failures** and provide fallbacks.
12. **Use Matrix power levels** for permission management.

-----

## 13\. Text Transformation and Replacement Patterns

### ❌ What Didn't Work

**Problem**: Manual string manipulation for text transformations.

  - Hardcoded string replacements that are difficult to maintain.
  - No support for complex pattern matching or conditional logic.
  - Inflexible response generation that requires code changes for new patterns.

**Problem**: Poor separation between pattern matching and response generation.

  - Mixing regex logic with response formatting in the same code.
  - Difficulty reusing patterns across different response types.
  - Hard to test pattern matching independently from response generation.

**Problem**: Limited template capabilities.

  - No support for dynamic content based on matched patterns.
  - Inability to handle complex data structures in responses.
  - Poor support for conditional content inclusion/exclusion.

### ✅ What Worked

**Solution**: Use the ReactBot pattern-template architecture for flexible text transformation and replacement.

### Core Architecture Patterns from ReactBot

**1. Rule-Based Pattern Matching**:
```python
# Rules define what to match and how to respond
@dataclass
class Rule:
    rooms: Set[RoomID]              # Room restrictions
    not_rooms: Set[RoomID]          # Room exclusions
    matches: List[RPattern]         # Patterns to match
    not_matches: List[RPattern]     # Patterns to exclude
    template: Template              # Response template
    type: Optional[EventType]       # Override event type
    variables: Dict[str, Any]       # Rule-specific variables

    def match(self, evt: MessageEvent) -> Optional[Match]:
        # Check room restrictions
        if len(self.rooms) > 0 and evt.room_id not in self.rooms:
            return None
        elif evt.room_id in self.not_rooms:
            return None
        
        # Check positive matches
        for pattern in self.matches:
            match = pattern.search(evt.content.body)
            if match:
                # Check negative matches
                if self._check_not_match(evt.content.body):
                    return None
                return match
        return None
```

**2. Template-Based Response Generation**:
```python
# Templates separate content structure from pattern matching
@dataclass
class Template:
    type: EventType                                    # Matrix event type
    variables: Dict[str, Any]                         # Template variables
    content: Union[Dict[str, Any], JinjaStringTemplate] # Content structure

    def execute(self, evt: Event, rule_vars: Dict[str, Any], extra_vars: Dict[str, str]) -> Dict[str, Any]:
        # Merge variables from multiple sources
        variables = extra_vars
        for name, template in chain(rule_vars.items(), self.variables.items()):
            if isinstance(template, JinjaNativeTemplate):
                # Render Jinja2 templates with full context
                rendered_var = template.render(event=evt, variables=variables, **global_vars)
                variables[name] = rendered_var
            else:
                variables[name] = template
        
        # Process content template
        if isinstance(self.content, JinjaStringTemplate):
            # Full Jinja2 template for complex logic
            raw_json = self.content.render(event=evt, **variables)
            return json.loads(raw_json)
        else:
            # Simple variable substitution in structured content
            content = copy.deepcopy(self.content)
            for path in self._variable_locations:
                # Replace $${variable} patterns with actual values
                self._replace_variables_at_path(content, path, variables)
            return content
```

**3. Variable Substitution System**:
```python
# Two-tier variable system: simple substitution and Jinja2 templates
variable_regex = re.compile(r"\$\${([0-9A-Za-z-_]+)}")

@staticmethod
def _replace_variables(tpl: str, variables: Dict[str, Any]) -> str:
    full_var_match = variable_regex.fullmatch(tpl)
    if full_var_match:
        # Whole field is a single variable, preserve type
        return variables[full_var_match.group(1)]
    # Partial substitution, convert to string
    return variable_regex.sub(lambda match: str(variables[match.group(1)]), tpl)
```

### Real-World Text Transformation Examples

**1. URL Transformation (Twitter to Nitter)**:
```yaml
# Configuration-based URL replacement
templates:
    nitter:
        type: m.room.message
        content:
            msgtype: m.text
            body: https://nitter.net/$${1}/status/$${2}

rules:
    twitter:
        matches:
        - https://twitter.com/(.+?)/status/(\d+)
        template: nitter
```

**2. Complex Text Replacement (Stallman Bot)**:
```yaml
# Large text replacements with context awareness
templates:
    plaintext_notice:
        type: m.room.message
        content:
            msgtype: m.notice
            body: $${message}

rules:
    linux:
        matches: [linux]
        not_matches: [gnu, kernel]  # Avoid false positives
        template: plaintext_notice
        variables:
            message: |
                I'd just like to interject for one moment. What you're referring to as Linux, 
                is in fact, GNU/Linux, or as I've recently taken to calling it, GNU plus Linux...
```

**3. Dynamic Reactions with Randomization**:
```yaml
# Random selection from predefined options
templates:
    random_reaction:
        type: m.reaction
        variables:
            react_to_event: '{{event.event_id}}'
            reaction: '{{ variables.reaction_choices | random }}'
        content:
            m.relates_to:
                rel_type: m.annotation
                event_id: $${react_to_event}
                key: $${reaction}

rules:
    random:
        matches: [hmm]
        template: random_reaction
        variables:
            reaction_choices: [🤔, 🧐, 🤨]
```

**4. Thread-Aware Responses**:
```yaml
# Conditional logic based on message context
templates:
    thread_or_reply:
        type: m.room.message
        variables:
            relates_to: |
                {{
                    {"rel_type": "m.thread", "event_id": event.content.get_thread_parent(), 
                     "is_falling_back": True, "m.in_reply_to": {"event_id": event.event_id}}
                    if event.content.get_thread_parent()
                    else {"m.in_reply_to": {"event_id": event.event_id}}
                }}
        content:
            msgtype: m.text
            body: $${text}
            m.relates_to: $${relates_to}
```

### Pattern Optimization Strategies

**1. Simple Pattern Optimization**:
```python
# ReactBot automatically optimizes simple patterns
class SimplePattern:
    @classmethod
    def compile(cls, pattern: str, flags: re.RegexFlag, raw: bool = None) -> Optional['SimplePattern']:
        # Convert simple regex patterns to faster string operations
        if raw is not False and (not flags & re.MULTILINE or raw is True):
            if pattern.startswith('^') and pattern.endswith('$'):
                # Exact match
                return cls(pattern[1:-1], 'exact')
            elif pattern.startswith('^'):
                # Starts with
                return cls(pattern[1:], 'startswith')
            elif pattern.endswith('$'):
                # Ends with
                return cls(pattern[:-1], 'endswith')
            elif not any(char in pattern for char in r'.*+?[]{}()|\^$'):
                # Contains (no special regex chars)
                return cls(pattern, 'contains')
        return None
```

**2. Anti-Spam Integration**:
```python
# Built-in rate limiting for text transformations
class ReactBot(Plugin):
    def is_flood(self, evt: MessageEvent) -> bool:
        # Room-based rate limiting
        room_key = f"room:{evt.room_id}"
        if self._check_rate_limit(room_key, self.config.antispam.room):
            return True
        
        # User-based rate limiting
        user_key = f"user:{evt.sender}"
        if self._check_rate_limit(user_key, self.config.antispam.user):
            return True
        
        return False
```

### Advanced Template Patterns

**1. Conditional Content with Omit Values**:
```python
# Use special OmitValue to conditionally exclude content
global_vars = {
    "omit": OmitValue,  # Special value to omit fields
}

# In template execution
if replaced_data is OmitValue:
    del data[key]  # Remove the field entirely
else:
    data[key] = replaced_data
```

**2. Multi-Source Variable Merging**:
```python
# Variables from multiple sources with precedence
def execute(self, evt: Event, rule_vars: Dict[str, Any], extra_vars: Dict[str, str]) -> Dict[str, Any]:
    variables = extra_vars  # Start with regex capture groups
    
    # Add rule-specific variables (override extra_vars)
    for name, template in chain(rule_vars.items(), self.variables.items()):
        if isinstance(template, JinjaNativeTemplate):
            # Dynamic variables with full template context
            rendered_var = template.render(event=evt, variables=variables, **global_vars)
            variables[name] = rendered_var
        else:
            # Static variables
            variables[name] = template
    
    return self._apply_variables_to_content(variables)
```

**3. Type-Preserving Variable Substitution**:
```python
# Preserve non-string types in variable substitution
@staticmethod
def _replace_variables(tpl: str, variables: Dict[str, Any]) -> Any:
    full_var_match = variable_regex.fullmatch(tpl)
    if full_var_match:
        # Whole field is a single variable, preserve original type
        return variables[full_var_match.group(1)]
    # Partial substitution, must convert to string
    return variable_regex.sub(lambda match: str(variables[match.group(1)]), tpl)
```

### Configuration Patterns

**1. Hierarchical Configuration Structure**:
```yaml
# Separate templates from rules for reusability
templates:
    reaction:
        type: m.reaction
        variables:
            react_to_event: "{{event.content.get_reply_to() or event.event_id}}"
        content:
            m.relates_to:
                rel_type: m.annotation
                event_id: $${react_to_event}
                key: $${reaction}

default_flags: [ignorecase]  # Global regex flags

antispam:  # Built-in rate limiting
    room: {max: 1, delay: 60}
    user: {max: 2, delay: 60}

rules:
    twim_cookies:
        rooms: ["!FPUfgzXYWTKgIrwKxW:matrix.org"]  # Room restrictions
        matches: [^TWIM]                            # Pattern to match
        template: reaction                          # Template to use
        variables: {reaction: 🍪}                   # Rule-specific variables
```

**2. Pattern Complexity Levels**:
```yaml
# Simple string matching
rules:
    simple:
        matches: [alot]  # Simple contains match
        template: alot_image

# Regex with capture groups
rules:
    regex:
        matches: [https://twitter.com/(.+?)/status/(\d+)]
        template: nitter  # Uses $${1} and $${2}

# Complex patterns with flags
rules:
    complex:
        matches:
        - pattern: (?P<user>\w+) said (?P<quote>.+)
          flags: [ignorecase, multiline]
        template: quote_response  # Uses $${user} and $${quote}
```

### Implementation Patterns for Other Plugins

**1. Integrating Text Transformation in Existing Plugins**:
```python
# Add transformation capability to any plugin
class MyPlugin(Plugin):
    def __init__(self):
        self.transformations = {}
        self.load_transformations()
    
    def load_transformations(self):
        # Load from config similar to ReactBot
        for name, config in self.config["transformations"].items():
            pattern = re.compile(config["pattern"], re.IGNORECASE)
            template = config["template"]
            self.transformations[name] = (pattern, template)
    
    @event.on(EventType.ROOM_MESSAGE)
    async def handle_message(self, evt: MessageEvent) -> None:
        for name, (pattern, template) in self.transformations.items():
            match = pattern.search(evt.content.body)
            if match:
                response = self.apply_template(template, match, evt)
                await evt.respond(response)
                break
```

**2. Template System for Command Responses**:
```python
# Use template patterns for command responses
class CommandPlugin(Plugin):
    @command.new("status")
    async def status_command(self, evt: MessageEvent) -> None:
        template_vars = {
            "user": evt.sender,
            "room": evt.room_id,
            "time": datetime.now().isoformat(),
            "uptime": self.get_uptime(),
        }
        
        response = self.render_template("status_response", template_vars)
        await evt.respond(response)
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        template = self.config["templates"][template_name]
        return template.format(**variables)
```

### 🔧 Standard Operating Procedure

1.  **Separate pattern matching from response generation** using rule-template architecture.
2.  **Use regex capture groups** to extract data from matched text (`$${1}`, `$${2}`, named groups).
3.  **Implement two-tier variable system**: simple `$${var}` substitution and Jinja2 `{{template}}` logic.
4.  **Optimize simple patterns** to string operations when possible (exact, startswith, endswith, contains).
5.  **Include anti-spam measures** with room and user-based rate limiting.
6.  **Use conditional content** with `OmitValue` to dynamically include/exclude fields.
7.  **Preserve data types** in variable substitution when the entire field is a single variable.
8.  **Structure configuration hierarchically** with reusable templates and specific rules.
9.  **Support negative matching** with `not_matches` to avoid false positives.
10. **Test patterns independently** from response generation for better maintainability.

-----

## 14\. Standard Operating Procedures

### Plugin Development Workflow

1.  **Define plugin purpose**: What problem does it solve or what feature does it add?
2.  **Choose architecture pattern**: Single-file, medium, or complex based on requirements.
3.  **Design `maubot.yaml`**: Use reverse domain naming, set appropriate flags (`database`, `webapp`, `config`).
4.  **Define `Config` class and `base-config.yaml`**: Use `BaseProxyConfig` with `helper.copy()` pattern.
5.  **Set up database migrations**: Use `UpgradeTable` with `@upgrade_table.register()` decorators.
6.  **Plan file organization**: Follow established patterns based on complexity.
7.  **Implement core logic**: Use established command/event handler patterns.
8.  **Add external service integration**: Use proper HTTP clients, authentication, and error handling.
9.  **Implement webhook handling**: With signature verification and event aggregation.
10. **Add Matrix message handling**: Support encryption, different message types, and security.
11. **Add background tasks**: Use `background_task.create()` and proper cancellation.
12. **Add comprehensive logging**: Use `self.log` at appropriate levels.
13. **Set up code quality**: Add `pyproject.toml`, `.pre-commit-config.yaml`, `.gitlab-ci.yml`.
14. **Write unit tests**: Cover core logic, edge cases, and error conditions.
15. **Build and deploy**: Use `build_plugin.py` script with version increment for proper releases.
16. **Iterate quickly**: Use `mbc build --upload` and test in a dev Matrix room.
17. **Refine and Document**: Clean up code, add comments, update documentation.

### Plugin Build and Release Process

**CRITICAL**: Always use the `build_plugin.py` script for building releases, not manual zip commands.

1.  **Version Management**:
    - **Always increment version in `maubot.yaml`** before building
    - Use semantic versioning (e.g., 1.0.9 → 1.0.10 for bug fixes, 1.0.10 → 1.1.0 for features)
    - Version must be incremented for each release to avoid conflicts

2.  **Build Process**:
    ```bash
    # Increment version in plugin/maubot.yaml first
    # Then build using the script
    python3 build_plugin.py -v
    ```
    
3.  **Build Script Benefits**:
    - Automatically includes all required files based on `maubot.yaml`
    - Generates proper filename with version (e.g., `may.irregularchat.matrix_to_discourse-v1.0.10.mbp`)
    - Validates plugin structure before building
    - Provides verbose output for debugging build issues

4.  **What Gets Included**:
    - All modules listed in `maubot.yaml`
    - `base-config.yaml` if config is enabled
    - Extra files specified in `extra_files`
    - Proper directory structure for multi-module plugins

5.  **Common Build Issues**:
    - **Missing modules**: Ensure all Python files are listed in `modules`
    - **Wrong main_class**: Must match actual class name and module structure
    - **Missing dependencies**: List all required packages in `dependencies`
    - **File not found**: Check `extra_files` paths are correct

### Critical Event Handler Issues

**CRITICAL**: Multiple event handlers for the same event type can cause conflicts and silent failures.

**Problem**: Having multiple `@event.on(EventType.ROOM_MESSAGE)` decorators in the same plugin class can cause only one handler to execute, leading to features silently failing.

**Example of problematic code**:
```python
@event.on(EventType.ROOM_MESSAGE)
async def handle_message(self, evt: MessageEvent) -> None:
    # URL processing logic
    pass

@event.on(EventType.ROOM_MESSAGE)  # CONFLICT!
async def handle_matrix_reply(self, evt: MessageEvent) -> None:
    # Reply handling logic
    pass
```

**Solution**: Use a single event handler and route internally:
```python
@event.on(EventType.ROOM_MESSAGE)
async def handle_message(self, evt: MessageEvent) -> None:
    # Handle URL processing
    if self.should_process_urls(evt):
        await self.process_urls(evt)
    
    # Handle replies
    if self.is_reply(evt):
        await self.handle_reply(evt)
```

**Debugging Tips**:
- Add comprehensive debug logging to see which handlers are actually executing
- Use single event handlers with internal routing for complex message processing
- Test all features after any event handler changes

### Debugging Workflow

1.  **Check Maubot UI logs first**: The fastest way to see immediate errors.
2.  **Use `mbc logs <plugin_id>`**: For more detailed CLI access to logs.
3.  **Add `self.log.debug()` statements**: Strategically place them around suspicious code.
4.  **Verify `maubot.yaml` and `base-config.yaml`**: Often, simple typos or incorrect paths here cause load failures.
5.  **Isolate the problem**: Temporarily comment out parts of the code to narrow down the source of the issue.
6.  **Run unit tests**: See if the issue is caught by existing tests, or write a new test to reproduce it.
7.  **Consult Maubot documentation and Matrix dev rooms**: For obscure errors, the Maubot docs or the `#maubot:maunium.net` Matrix room are excellent resources.

### Code Quality Checklist

  - [ ] `maubot.yaml` follows established patterns (reverse domain ID, correct flags).
  - [ ] Plugin uses `BaseProxyConfig` for configuration with `helper.copy()` pattern.
  - [ ] Database migrations use `UpgradeTable` with `@classmethod get_db_upgrade_table()`.
  - [ ] `async def start()` and `async def stop()` are implemented for resource management.
  - [ ] All I/O operations are `await`ed.
  - [ ] Background tasks use `background_task.create()` and are properly cancelled.
  - [ ] Commands use `@command.new()` with `@command.argument()` for parameters.
  - [ ] Event handling uses appropriate decorators (`@event.on()`, `@command.passive()`).
  - [ ] `self.log` is used consistently for all output.
  - [ ] `try...except` blocks are specific and handle potential errors gracefully.
  - [ ] Database access uses parameterized queries (`$1`, `$2`, etc.).
  - [ ] Code quality tools are configured (`pyproject.toml`, `.pre-commit-config.yaml`).
  - [ ] CI/CD is configured appropriately (GitHub Actions for GitHub, GitLab CI for GitLab).
  - [ ] Unit tests exist for critical components.
  - [ ] Bot accounts have necessary Matrix power levels for administrative commands.
  - [ ] Sensitive information is handled via configuration, not hardcoded.
  - [ ] **Version incremented in `maubot.yaml` before each build**.
  - [ ] **Plugin built using `build_plugin.py` script, not manual zip commands**.

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

1.  **Follow established patterns**: The maubot ecosystem has well-defined conventions - use them.
2.  **Architecture matters**: Choose single-file, medium, or complex patterns based on actual needs.
3.  **Configuration standardization**: Use `BaseProxyConfig` with `helper.copy()` for robust config management.
4.  **Database migrations are essential**: Use `UpgradeTable` with proper versioning from the start.
5.  **Background tasks**: Use `background_task.create()` and `AsyncIOScheduler` for complex scheduling.
6.  **Command handling**: Leverage `@command.argument()`, subcommands, and passive pattern matching.
7.  **Code quality is standard**: Use `pyproject.toml`, pre-commit hooks, and shared CI/CD.
8.  **Error handling patterns**: Specific exception handling with user-friendly messages and comprehensive logging.
9.  **Testing strategy**: Unit tests with mocked dependencies, integration tests in dedicated environments.
10. **Deployment automation**: `mbc build --upload` for rapid iteration, proper versioning for releases.

**Real-world insights from analyzed plugins**:
- Simple plugins (timer) can be very effective with minimal code
- Medium plugins (faqbot, reactbot) benefit from modular organization
- Complex plugins (reminder, github) require careful architecture and separation of concerns
- All successful plugins follow similar patterns for configuration, database, and command handling
- Consistent tooling and CI/CD across the ecosystem reduces maintenance burden

This document should be updated as new lessons are learned during continued Maubot plugin development.
