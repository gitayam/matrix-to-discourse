"""Database module for MatrixToDiscourseBot with migration support."""

from mautrix.util.async_db import UpgradeTable, Connection
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Create upgrade table for database migrations
upgrade_table = UpgradeTable()

@upgrade_table.register(description="Initial revision - message ID mappings")
async def upgrade_v1(conn: Connection) -> None:
    """Create initial database schema for message ID mappings."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS message_mappings (
            matrix_event_id TEXT PRIMARY KEY,
            discourse_topic_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_discourse_topic 
        ON message_mappings(discourse_topic_id)
    """)

@upgrade_table.register(description="Add room ID to mappings")
async def upgrade_v2(conn: Connection) -> None:
    """Add room_id column to track which room the message came from."""
    await conn.execute("""
        ALTER TABLE message_mappings 
        ADD COLUMN room_id TEXT
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_room_id 
        ON message_mappings(room_id)
    """)

@upgrade_table.register(description="Add user tracking")
async def upgrade_v3(conn: Connection) -> None:
    """Add user tracking for rate limiting and statistics."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            user_id TEXT NOT NULL,
            room_id TEXT NOT NULL,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            post_count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, room_id)
        )
    """)

class MessageMappingDatabase:
    """Database handler for message ID mappings."""
    
    def __init__(self, db):
        self.db = db
        
    async def store_mapping(self, matrix_event_id: str, discourse_topic_id: str, room_id: str) -> None:
        """Store a mapping between Matrix event ID and Discourse topic ID."""
        await self.db.execute("""
            INSERT INTO message_mappings (matrix_event_id, discourse_topic_id, room_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (matrix_event_id) DO UPDATE 
            SET discourse_topic_id = excluded.discourse_topic_id,
                room_id = excluded.room_id
        """, matrix_event_id, discourse_topic_id, room_id)
        
    async def get_mapping(self, matrix_event_id: str) -> Optional[str]:
        """Get Discourse topic ID for a Matrix event ID."""
        return await self.db.fetchval("""
            SELECT discourse_topic_id 
            FROM message_mappings 
            WHERE matrix_event_id = $1
        """, matrix_event_id)
        
    async def load_all_mappings(self) -> Dict[str, str]:
        """Load all message mappings from database."""
        rows = await self.db.fetch("""
            SELECT matrix_event_id, discourse_topic_id 
            FROM message_mappings
        """)
        return {row['matrix_event_id']: row['discourse_topic_id'] for row in rows}
        
    async def delete_mapping(self, matrix_event_id: str) -> None:
        """Delete a message mapping."""
        await self.db.execute("""
            DELETE FROM message_mappings 
            WHERE matrix_event_id = $1
        """, matrix_event_id)
        
    async def update_user_activity(self, user_id: str, room_id: str) -> None:
        """Update user activity for rate limiting."""
        await self.db.execute("""
            INSERT INTO user_activity (user_id, room_id, last_activity, post_count)
            VALUES ($1, $2, CURRENT_TIMESTAMP, 1)
            ON CONFLICT (user_id, room_id) DO UPDATE
            SET last_activity = CURRENT_TIMESTAMP,
                post_count = user_activity.post_count + 1
        """, user_id, room_id)
        
    async def get_user_post_count(self, user_id: str, room_id: str, minutes: int = 60) -> int:
        """Get user post count within the last N minutes."""
        return await self.db.fetchval("""
            SELECT COUNT(*) 
            FROM message_mappings 
            WHERE room_id = $1 
            AND matrix_event_id IN (
                SELECT matrix_event_id 
                FROM message_mappings 
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '%s minutes'
            )
        """, room_id, minutes) or 0 