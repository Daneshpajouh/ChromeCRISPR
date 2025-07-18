"""
Hierarchical Rate Limiter
Multi-API rate limiting with Redis-based token buckets and adaptive backoff
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an API"""
    requests_per_second: float
    burst_limit: int
    recovery_timeout: float
    backoff_multiplier: float = 1.0
    consecutive_failures: int = 0


class HierarchicalRateLimiter:
    """
    Hierarchical rate limiter with Redis-based token buckets
    Provides optimal throughput while respecting individual API constraints
    """

    def __init__(self, api_configs: Dict[str, Dict[str, Any]], redis_client=None):
        """
        Initialize rate limiter with API configurations

        Args:
            api_configs: Dictionary of API configurations
            redis_client: Optional Redis client for distributed rate limiting
        """
        self.redis_client = redis_client
        self.rate_limits = {}
        self.last_request_times = {}
        self.request_counts = {}
        self.failure_counts = {}
        self.backoff_multipliers = {}

        # Initialize rate limits from config
        for api_name, config in api_configs.items():
            self.rate_limits[api_name] = RateLimitConfig(
                requests_per_second=config.get('max_requests_per_second', 1.0),
                burst_limit=config.get('burst_limit', 10),
                recovery_timeout=config.get('recovery_timeout', 60.0),
                backoff_multiplier=1.0,
                consecutive_failures=0
            )
            self.last_request_times[api_name] = None
            self.request_counts[api_name] = 0
            self.failure_counts[api_name] = 0
            self.backoff_multipliers[api_name] = 1.0

        logger.info(f"Initialized rate limiter for {len(self.rate_limits)} APIs")

    async def acquire_token(self, api_name: str) -> bool:
        """
        Acquire rate limit token using distributed token bucket

        Args:
            api_name: Name of the API

        Returns:
            True if token acquired, False if rate limited
        """
        if api_name not in self.rate_limits:
            logger.warning(f"Unknown API: {api_name}")
            return True

        config = self.rate_limits[api_name]
        current_time = time.time()

        # Check if we're in backoff period
        if self._is_in_backoff(api_name, current_time):
            wait_time = self._calculate_backoff_wait_time(api_name)
            logger.debug(f"Rate limiter backoff for {api_name}: {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            return False

        # Calculate time since last request
        if self.last_request_times[api_name]:
            time_since_last = current_time - self.last_request_times[api_name]
            min_interval = 1.0 / config.requests_per_second

            # Apply backoff multiplier
            adjusted_interval = min_interval * self.backoff_multipliers[api_name]

            if time_since_last < adjusted_interval:
                wait_time = adjusted_interval - time_since_last
                logger.debug(f"Rate limiting {api_name}: sleeping for {wait_time:.3f}s")
                await asyncio.sleep(wait_time)

        # Update request tracking
        self.last_request_times[api_name] = current_time
        self.request_counts[api_name] += 1

        # Reset consecutive failures on successful request
        if self.failure_counts[api_name] > 0:
            self.failure_counts[api_name] = max(0, self.failure_counts[api_name] - 1)
            self.backoff_multipliers[api_name] = max(1.0, self.backoff_multipliers[api_name] * 0.8)

        return True

    async def acquire_token_redis(self, api_name: str) -> bool:
        """
        Acquire token using Redis-based distributed token bucket

        Args:
            api_name: Name of the API

        Returns:
            True if token acquired, False if rate limited
        """
        if not self.redis_client:
            return await self.acquire_token(api_name)

        config = self.rate_limits[api_name]
        current_time = time.time()

        # Redis key for this API's token bucket
        bucket_key = f"rate_limit:{api_name}:tokens"
        last_refill_key = f"rate_limit:{api_name}:last_refill"

        try:
            # Get current token count and last refill time
            async with self.redis_client.pipeline() as pipe:
                await pipe.get(bucket_key)
                await pipe.get(last_refill_key)
                results = await pipe.execute()

            current_tokens = float(results[0]) if results[0] else config.burst_limit
            last_refill = float(results[1]) if results[1] else current_time

            # Calculate tokens to add since last refill
            time_passed = current_time - last_refill
            tokens_to_add = time_passed * config.requests_per_second

            # Refill bucket (up to burst limit)
            new_tokens = min(config.burst_limit, current_tokens + tokens_to_add)

            if new_tokens >= 1.0:
                # Consume one token
                async with self.redis_client.pipeline() as pipe:
                    await pipe.set(bucket_key, new_tokens - 1.0)
                    await pipe.set(last_refill_key, current_time)
                    await pipe.execute()

                return True
            else:
                # Not enough tokens, calculate wait time
                wait_time = (1.0 - new_tokens) / config.requests_per_second
                logger.debug(f"Redis rate limit for {api_name}: waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
                return False

        except Exception as e:
            logger.error(f"Redis rate limiting error for {api_name}: {e}")
            # Fallback to local rate limiting
            return await self.acquire_token(api_name)

    def handle_rate_limit_error(self, api_name: str):
        """Handle rate limit errors with exponential backoff"""
        if api_name not in self.rate_limits:
            return

        config = self.rate_limits[api_name]
        config.consecutive_failures += 1
        config.backoff_multiplier = min(10.0, 2 ** config.consecutive_failures)

        logger.warning(f"Rate limit hit for {api_name}, increasing backoff to {config.backoff_multiplier}x")

    def handle_success(self, api_name: str):
        """Reset backoff on successful requests"""
        if api_name not in self.rate_limits:
            return

        config = self.rate_limits[api_name]
        if config.consecutive_failures > 0:
            config.consecutive_failures = max(0, config.consecutive_failures - 1)
            config.backoff_multiplier = max(1.0, config.backoff_multiplier * 0.8)

    def _is_in_backoff(self, api_name: str, current_time: float) -> bool:
        """Check if API is in backoff period"""
        config = self.rate_limits[api_name]

        if config.consecutive_failures == 0:
            return False

        # Check if enough time has passed since last failure
        if self.last_request_times[api_name]:
            time_since_failure = current_time - self.last_request_times[api_name]
            return time_since_failure < config.recovery_timeout

        return False

    def _calculate_backoff_wait_time(self, api_name: str) -> float:
        """Calculate wait time for backoff period"""
        config = self.rate_limits[api_name]
        base_interval = 1.0 / config.requests_per_second
        return base_interval * config.backoff_multiplier

    def get_rate_limit_info(self, api_name: str) -> Dict[str, Any]:
        """Get current rate limit information for an API"""
        if api_name not in self.rate_limits:
            return {}

        config = self.rate_limits[api_name]
        return {
            'requests_per_second': config.requests_per_second,
            'burst_limit': config.burst_limit,
            'backoff_multiplier': config.backoff_multiplier,
            'consecutive_failures': config.consecutive_failures,
            'total_requests': self.request_counts[api_name],
            'last_request_time': self.last_request_times[api_name],
            'in_backoff': self._is_in_backoff(api_name, time.time())
        }

    def get_all_rate_limit_info(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limit information for all APIs"""
        return {
            api_name: self.get_rate_limit_info(api_name)
            for api_name in self.rate_limits.keys()
        }

    async def reset_rate_limits(self, api_name: Optional[str] = None):
        """Reset rate limits for specific API or all APIs"""
        if api_name:
            if api_name in self.rate_limits:
                self.rate_limits[api_name].consecutive_failures = 0
                self.rate_limits[api_name].backoff_multiplier = 1.0
                self.last_request_times[api_name] = None
                logger.info(f"Reset rate limits for {api_name}")
        else:
            for api in self.rate_limits:
                self.rate_limits[api].consecutive_failures = 0
                self.rate_limits[api].backoff_multiplier = 1.0
                self.last_request_times[api] = None
            logger.info("Reset rate limits for all APIs")

    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Rate limiter closed")


class AdaptiveRateLimiter(HierarchicalRateLimiter):
    """
    Enhanced rate limiter with adaptive learning capabilities
    """

    def __init__(self, api_configs: Dict[str, Dict[str, Any]], redis_client=None):
        super().__init__(api_configs, redis_client)
        self.response_times = {}
        self.success_rates = {}
        self.adaptive_thresholds = {}

        # Initialize adaptive tracking
        for api_name in self.rate_limits:
            self.response_times[api_name] = []
            self.success_rates[api_name] = 1.0
            self.adaptive_thresholds[api_name] = self.rate_limits[api_name].requests_per_second

    def update_performance_metrics(self, api_name: str, response_time: float, success: bool):
        """Update performance metrics for adaptive learning"""
        if api_name not in self.rate_limits:
            return

        # Update response time tracking
        self.response_times[api_name].append(response_time)
        if len(self.response_times[api_name]) > 100:
            self.response_times[api_name].pop(0)

        # Update success rate
        current_rate = self.success_rates[api_name]
        if success:
            self.success_rates[api_name] = min(1.0, current_rate + 0.01)
        else:
            self.success_rates[api_name] = max(0.0, current_rate - 0.05)

        # Adaptive threshold adjustment
        self._adjust_adaptive_threshold(api_name)

    def _adjust_adaptive_threshold(self, api_name: str):
        """Adjust rate limit threshold based on performance"""
        config = self.rate_limits[api_name]
        success_rate = self.success_rates[api_name]

        # Calculate average response time
        if self.response_times[api_name]:
            avg_response_time = sum(self.response_times[api_name]) / len(self.response_times[api_name])
        else:
            avg_response_time = 1.0

        # Adjust threshold based on success rate and response time
        if success_rate > 0.95 and avg_response_time < 2.0:
            # Good performance, can increase rate
            new_threshold = min(config.requests_per_second * 1.2, config.requests_per_second * 2)
        elif success_rate < 0.8 or avg_response_time > 5.0:
            # Poor performance, decrease rate
            new_threshold = max(config.requests_per_second * 0.5, config.requests_per_second * 0.8)
        else:
            # Stable performance, maintain current rate
            new_threshold = config.requests_per_second

        self.adaptive_thresholds[api_name] = new_threshold
        logger.debug(f"Adaptive threshold for {api_name}: {new_threshold:.2f} req/s")

    async def acquire_token(self, api_name: str) -> bool:
        """Override to use adaptive thresholds"""
        if api_name in self.adaptive_thresholds:
            # Temporarily use adaptive threshold
            original_rate = self.rate_limits[api_name].requests_per_second
            self.rate_limits[api_name].requests_per_second = self.adaptive_thresholds[api_name]

            result = await super().acquire_token(api_name)

            # Restore original rate
            self.rate_limits[api_name].requests_per_second = original_rate

            return result

        return await super().acquire_token(api_name)
