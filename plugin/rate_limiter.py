"""Rate limiting module for MatrixToDiscourseBot."""

import time
from typing import Dict, List
from mautrix.types import UserID
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementation based on ReactBot patterns."""
    
    def __init__(self, max_requests: int, window_minutes: int):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: Dict[str, List[float]] = {}  # key -> list of timestamps
        
    def check_rate_limit(self, key: str) -> bool:
        """
        Check if a request should be rate limited.
        
        Args:
            key: The rate limit key (e.g., "user:@user:example.com" or "room:!room:example.com")
            
        Returns:
            True if rate limited, False if allowed
        """
        now = time.time()
        request_times = self.requests.get(key, [])
        
        # Remove old requests outside the window
        request_times = [req_time for req_time in request_times 
                        if now - req_time < self.window_seconds]
        
        if len(request_times) >= self.max_requests:
            logger.debug(f"Rate limit hit for {key}: {len(request_times)}/{self.max_requests} in {self.window_seconds}s")
            return True  # Rate limited
        
        # Add current request
        request_times.append(now)
        self.requests[key] = request_times
        return False
        
    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        if key in self.requests:
            del self.requests[key]
            
    def cleanup_old_entries(self) -> None:
        """Remove old entries to prevent memory growth."""
        now = time.time()
        keys_to_remove = []
        
        for key, request_times in self.requests.items():
            # Remove old requests
            request_times = [req_time for req_time in request_times 
                           if now - req_time < self.window_seconds]
            
            if not request_times:
                keys_to_remove.append(key)
            else:
                self.requests[key] = request_times
                
        for key in keys_to_remove:
            del self.requests[key]
            
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} expired rate limit entries")

class AntiSpamConfig:
    """Anti-spam configuration based on ReactBot patterns."""
    
    def __init__(self, config: dict):
        # Room-based limits
        room_config = config.get("antispam", {}).get("room", {})
        self.room_max = room_config.get("max", 5)
        self.room_delay = room_config.get("delay", 60)
        
        # User-based limits
        user_config = config.get("antispam", {}).get("user", {})
        self.user_max = user_config.get("max", 10)
        self.user_delay = user_config.get("delay", 60)
        
        # Create rate limiters
        self.room_limiter = RateLimiter(self.room_max, self.room_delay)
        self.user_limiter = RateLimiter(self.user_max, self.user_delay)
        
    def is_flood(self, user_id: UserID, room_id: str) -> bool:
        """
        Check if a message should be considered flood/spam.
        
        Args:
            user_id: The user sending the message
            room_id: The room where the message is sent
            
        Returns:
            True if the message should be blocked, False otherwise
        """
        # Check room rate limit
        room_key = f"room:{room_id}"
        if self.room_limiter.check_rate_limit(room_key):
            return True
            
        # Check user rate limit
        user_key = f"user:{user_id}"
        if self.user_limiter.check_rate_limit(user_key):
            return True
            
        return False
        
    def cleanup(self) -> None:
        """Clean up old rate limit entries."""
        self.room_limiter.cleanup_old_entries()
        self.user_limiter.cleanup_old_entries() 