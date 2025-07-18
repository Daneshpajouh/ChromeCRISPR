"""
Base API Client for GeneX Phase 1
Provides rate limiting, error handling, and common functionality for all API clients.
Enhanced with advanced features for comprehensive data mining.
"""

import time
import logging
import requests
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import yaml
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from collections import defaultdict, Counter
import aiohttp
import yarl
from urllib.parse import urlencode

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response wrapper"""
    data: Any
    status_code: int
    headers: Dict[str, str]
    url: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class APIMetrics:
    """Comprehensive API performance and usage metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    rate_limit_hits: int = 0
    retry_counts: Dict[int, int] = field(default_factory=dict)
    last_request_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)

    def update_metrics(self, response: APIResponse, response_time: float):
        """Update metrics with response data"""
        self.total_requests += 1
        self.response_times.append(response_time)

        if response.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            error_type = type(response.error_message).__name__ if response.error_message else 'Unknown'
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        if response.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if response.retry_count > 0:
            self.retry_counts[response.retry_count] = self.retry_counts.get(response.retry_count, 0) + 1

        if response.rate_limit_info:
            self.rate_limit_hits += 1

        self.last_request_time = datetime.now()
        self.average_response_time = statistics.mean(self.response_times) if self.response_times else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            'cache_hit_rate': self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            'average_response_time': self.average_response_time,
            'min_response_time': min(self.response_times) if self.response_times else 0,
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'error_distribution': dict(self.error_counts),
            'retry_distribution': dict(self.retry_counts),
            'rate_limit_hits': self.rate_limit_hits,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }

class RateLimiter:
    """Enhanced rate limiter with adaptive backoff and burst handling"""

    def __init__(self, max_requests_per_second: float, burst_limit: int = 10):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.burst_limit = burst_limit
        self.last_request_time = None
        self.request_count = 0
        self.burst_count = 0
        self.reset_time = datetime.now() + timedelta(hours=1)
        self.backoff_multiplier = 1.0
        self.consecutive_failures = 0

    def wait_if_needed(self):
        """Enhanced wait logic with adaptive backoff"""
        current_time = time.time()

        if self.last_request_time:
            elapsed = current_time - self.last_request_time
            adjusted_interval = self.min_interval * self.backoff_multiplier

            if elapsed < adjusted_interval:
                sleep_time = adjusted_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s (backoff: {self.backoff_multiplier}x)")
                time.sleep(sleep_time)

        self.last_request_time = current_time
        self.request_count += 1
        self.burst_count += 1

        # Reset burst count if enough time has passed
        if self.burst_count >= self.burst_limit:
            self.burst_count = 0
            time.sleep(self.min_interval * 2)  # Extra delay after burst

    def handle_rate_limit_error(self):
        """Handle rate limit errors with exponential backoff"""
        self.consecutive_failures += 1
        self.backoff_multiplier = min(10.0, 2 ** self.consecutive_failures)
        logger.warning(f"Rate limit hit, increasing backoff to {self.backoff_multiplier}x")

    def handle_success(self):
        """Reset backoff on successful requests"""
        if self.consecutive_failures > 0:
            self.consecutive_failures = max(0, self.consecutive_failures - 1)
            self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.8)

class CacheManager:
    """Enhanced cache manager with compression and metadata tracking"""

    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()

    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata for tracking"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
        return {'entries': {}, 'total_size': 0, 'last_cleanup': datetime.now().isoformat()}

    def _save_cache_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache metadata: {e}")

    def get_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key from URL and parameters"""
        cache_data = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        cache_key = self.get_cache_key(url, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                # Check if cache is still valid (24 hours)
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=24):
                    logger.debug(f"Cache hit for {url}")
                    return cached_data['data']
                else:
                    # Remove expired cache
                    cache_file.unlink()
                    self._update_cache_metadata(cache_key, 0, remove=True)
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")

        return None

    def set(self, url: str, params: Dict[str, Any], data: Dict[str, Any]):
        """Cache response data with size tracking"""
        cache_key = self.get_cache_key(url, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'url': url,
                'params': params
            }

            # Check cache size before writing
            self._cleanup_if_needed()

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)

            file_size = cache_file.stat().st_size
            self._update_cache_metadata(cache_key, file_size)

            logger.debug(f"Cached response for {url} ({file_size} bytes)")
        except Exception as e:
            logger.warning(f"Error caching response: {e}")

    def _update_cache_metadata(self, cache_key: str, size: int, remove: bool = False):
        """Update cache metadata"""
        if remove:
            if cache_key in self.cache_metadata['entries']:
                self.cache_metadata['total_size'] -= self.cache_metadata['entries'][cache_key]['size']
                del self.cache_metadata['entries'][cache_key]
        else:
            self.cache_metadata['entries'][cache_key] = {
                'size': size,
                'timestamp': datetime.now().isoformat()
            }
            self.cache_metadata['total_size'] += size

        self._save_cache_metadata()

    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds size limit"""
        if self.cache_metadata['total_size'] > self.max_cache_size:
            logger.info("Cache size limit exceeded, cleaning up...")

            # Sort entries by timestamp (oldest first)
            entries = sorted(
                self.cache_metadata['entries'].items(),
                key=lambda x: x[1]['timestamp']
            )

            # Remove oldest entries until under limit
            for cache_key, entry_info in entries:
                if self.cache_metadata['total_size'] <= self.max_cache_size * 0.8:
                    break

                cache_file = self.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    cache_file.unlink()

                self.cache_metadata['total_size'] -= entry_info['size']
                del self.cache_metadata['entries'][cache_key]

            self.cache_metadata['last_cleanup'] = datetime.now().isoformat()
            self._save_cache_metadata()

    def get_cache_size(self) -> int:
        """Return the number of items in the cache."""
        return len(self.cache_metadata['entries']) if hasattr(self, 'cache_metadata') else 0

class BaseAPIClient:
    """
    Enterprise-grade base API client with async-first design and comprehensive resilience.

    Implements:
    - Robust URL construction with yarl
    - Singleton aiohttp.ClientSession management
    - Adaptive rate limiting with token bucket
    - Exponential backoff retry logic
    - Circuit breaker pattern
    - Comprehensive error handling
    """

    def __init__(self, config: Dict[str, Any], session: Optional[aiohttp.ClientSession] = None):
        """
        Initialize the base API client with configuration and optional session.

        Args:
            config: Configuration dictionary containing base_url, rate_limit, etc.
            session: Optional aiohttp.ClientSession for dependency injection
        """
        # Validate and set base URL using yarl
        base_url = config.get('base_url', '')
        if not base_url:
            raise ValueError("base_url is required in config")

        # Ensure base_url ends with '/' for proper path joining
        if not base_url.endswith('/'):
            base_url += '/'

        self.base_url = yarl.URL(base_url)
        self.session = session

        # Rate limiting configuration
        self.rate_limit_per_sec = config.get('rate_limit_per_sec', 1.0)
        self.rate_limit_semaphore = asyncio.Semaphore(int(self.rate_limit_per_sec))

        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.max_retry_delay = config.get('max_retry_delay', 60.0)

        # Circuit breaker configuration
        self.failure_threshold = config.get('failure_threshold', 5)
        self.recovery_timeout = config.get('recovery_timeout', 60)
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

        # API key and authentication
        self.api_key = config.get('api_key')
        self.auth_header = config.get('auth_header', 'Authorization')

        logger.info(f"Initialized {self.__class__.__name__} with base_url: {self.base_url}")

    def _build_url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> yarl.URL:
        """
        Build a complete URL using yarl for safe path joining and query parameter encoding.

        Args:
            endpoint: API endpoint path (e.g., 'esearch.fcgi')
            params: Query parameters as dictionary

        Returns:
            Complete yarl.URL object
        """
        # Remove leading slash from endpoint to avoid double slashes
        endpoint = endpoint.lstrip('/')

        # Join base URL with endpoint using yarl's / operator
        url = self.base_url / endpoint

        # Add query parameters if provided
        if params:
            # Filter out None values
            clean_params = {k: v for k, v in params.items() if v is not None}
            if clean_params:
                url = url.with_query(clean_params)

        logger.debug(f"Built URL: {url}")
        return url

    async def _make_request(self, method: str, endpoint: str,
                          params: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None,
                          **kwargs) -> APIResponse:
        """
        Make an HTTP request with comprehensive error handling and retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            headers: Request headers
            **kwargs: Additional arguments for aiohttp

        Returns:
            APIResponse object with standardized response data
        """
        # Check circuit breaker state
        if self.circuit_state == 'OPEN':
            if self._should_attempt_reset():
                self.circuit_state = 'HALF_OPEN'
                logger.info(f"Circuit breaker for {self.__class__.__name__} entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN for {self.__class__.__name__}")

        # Apply rate limiting
        async with self.rate_limit_semaphore:
            # Build the complete URL
            url = self._build_url(endpoint, params)

            # Prepare headers
            request_headers = headers or {}
            if self.api_key:
                if self.auth_header == 'Authorization':
                    request_headers['Authorization'] = f'Bearer {self.api_key}'
                else:
                    request_headers[self.auth_header] = self.api_key

            # Make request with retry logic
            for attempt in range(self.max_retries + 1):
                try:
                    if not self.session:
                        raise Exception("No aiohttp session available")

                    start_time = time.time()

                    async with self.session.request(
                        method, str(url), headers=request_headers, **kwargs
                    ) as response:
                        duration = time.time() - start_time

                        # Parse response
                        if response.content_type == 'application/json':
                            data = await response.json()
                        else:
                            data = await response.text()

                        # Check for rate limiting
                        if response.status == 429:
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                wait_time = int(retry_after)
                                logger.warning(f"Rate limited by {self.__class__.__name__}, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                            else:
                                wait_time = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                                logger.warning(f"Rate limited by {self.__class__.__name__}, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                            continue

                        # Success - reset circuit breaker
                        if self.circuit_state != 'CLOSED':
                            self.circuit_state = 'CLOSED'
                            self.failure_count = 0
                            logger.info(f"Circuit breaker for {self.__class__.__name__} reset to CLOSED")

                        logger.debug(f"{self.__class__.__name__} {method} {endpoint} - {response.status} ({duration:.2f}s)")

                        return APIResponse(
                            data=data,
                            status_code=response.status,
                            headers=dict(response.headers),
                            url=str(url),
                            success=200 <= response.status < 300
                        )

                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    logger.warning(f"{self.__class__.__name__} request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

                    # Check if circuit breaker should open
                    if self.failure_count >= self.failure_threshold:
                        self.circuit_state = 'OPEN'
                        logger.error(f"Circuit breaker for {self.__class__.__name__} opened after {self.failure_count} failures")

                    # If this was the last attempt, raise the exception
                    if attempt == self.max_retries:
                        return APIResponse(
                            data=None,
                            status_code=0,
                            headers={},
                            url=str(url),
                            success=False,
                            error_message=str(e)
                        )

                    # Wait before retry with exponential backoff
                    wait_time = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                    await asyncio.sleep(wait_time)

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Make a GET request"""
        return await self._make_request('GET', endpoint, params=params, **kwargs)

    async def post(self, endpoint: str, data: Optional[Any] = None,
                  params: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Make a POST request"""
        return await self._make_request('POST', endpoint, data=data, params=params, **kwargs)

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure > self.recovery_timeout

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive client metrics"""
        return {
            'api_name': self.__class__.__name__,
            'metrics': self.metrics.get_summary(),
            'rate_limiter': {
                'max_requests_per_second': self.rate_limit_per_sec,
                'backoff_multiplier': self.rate_limiter.backoff_multiplier,
                'consecutive_failures': self.rate_limiter.consecutive_failures
            },
            'cache': {
                'total_entries': len(self.cache.cache_metadata['entries']),
                'total_size_mb': self.cache.cache_metadata['total_size'] / (1024 * 1024),
                'last_cleanup': self.cache.cache_metadata['last_cleanup']
            }
        }

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        return {
            'max_requests_per_second': self.rate_limit_per_sec,
            'backoff_multiplier': self.rate_limiter.backoff_multiplier,
            'consecutive_failures': self.rate_limiter.consecutive_failures,
            'last_request_time': self.rate_limiter.last_request_time
        }

    async def close(self):
        """Clean up resources"""
        self.session.close()
        logger.info(f"Closed {self.__class__.__name__} client")

    def get_cache_size(self) -> int:
        """Return the number of items in the cache."""
        if hasattr(self, 'cache_manager'):
            return self.cache_manager.get_cache_size() if hasattr(self.cache_manager, 'get_cache_size') else 0
        return 0

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with enhanced error handling"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required configuration
        required_sections = ['api_clients', 'logging', 'cache']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")

        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise

def create_client(api_name: str, config: Optional[Dict[str, Any]] = None) -> BaseAPIClient:
    """Create API client instance with enhanced error handling"""
    if config is None:
        config = load_config()

    api_config = config.get('api_clients', {}).get(api_name, {})

    if not api_config:
        raise ValueError(f"No configuration found for API client: {api_name}")

    # Import and create client based on name
    if api_name == 'pubmed':
        from .pubmed_client import PubMedClient
        return PubMedClient(api_config)
    elif api_name == 'semantic_scholar':
        from .semantic_scholar_client import SemanticScholarClient
        return SemanticScholarClient(api_config)
    elif api_name == 'crossref':
        from .crossref_client import CrossRefClient
        return CrossRefClient(api_config)
    elif api_name == 'ncbi':
        from .ncbi_client import NCBIClient
        return NCBIClient(api_config)
    elif api_name == 'ensembl':
        from .ensembl_client import EnsemblClient
        return EnsemblClient(api_config)
    else:
        raise ValueError(f"Unknown API client: {api_name}")
