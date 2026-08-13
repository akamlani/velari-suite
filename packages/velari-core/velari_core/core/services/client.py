import  httpx
import  logging
from    typing import Any, Dict, Optional, Self

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json",
}


class HttpClient(object):
    """Generic, reusable HTTP client — GET/POST with consistent logging and error handling.

    Usable immediately after construction; `connect()` is optional. Also works as a context
    manager, closing the session on exit.

    Args:
        base_url (Optional[str]): Prefix joined with every `get()`/`post()` path; leave unset
            to pass full URLs directly.
        headers (Optional[Dict[str, str]]): Merged over the default JSON `Content-Type`/
            `Accept` headers; overrides by key.
        follow_redirects (bool): Follow HTTP redirects; defaults to `True`.

    Examples:
        >>> client = HttpClient(base_url="https://registry.internal")
        >>> response = client.get("/datasets/churn-2026")
        >>> response.json()["name"]
        'churn-2026'
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        follow_redirects: bool = True,
    ) -> None:
        self.base_url = base_url or ""
        self.headers: Dict[str, str] = {**_DEFAULT_HEADERS, **(headers or {})}
        self._client = httpx.Client(headers=self.headers, follow_redirects=follow_redirects)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying session. Safe to call even after a failed `connect()`."""
        self._client.close()

    def connect(self, timeout: int = 30, **kwargs: Any) -> Self:
        """Open an HTTP/2 session against `base_url` and verify it's reachable via `/health`.

        Swaps in the new session only on success; a failed check leaves the prior one usable.

        Args:
            timeout (int): Request timeout in seconds for the new session; defaults to `30`.
            **kwargs (Any): Extra kwargs forwarded to `httpx.Client()`.

        Returns:
            Self: This client, so calls chain: `HttpClient(base_url=...).connect()`.

        Raises:
            httpx.HTTPError: If the health check fails (connection error, timeout, or HTTP error status).
        """
        try:
            client = httpx.Client(http2=True, headers=self.headers, timeout=timeout, **kwargs)
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"connect failed — {self.base_url}: {e}")
            raise
        self._client = client
        logger.info(f"Connected to {self.base_url}")
        return self

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET `base_url` + `url`, raising on any connection/timeout/HTTP-status failure.

        Args:
            url (str): Path to fetch, joined onto `base_url` (or a full URL if unset).
            **kwargs (Any): Extra kwargs forwarded to `httpx.Client.get()`.

        Returns:
            httpx.Response: The response, after `raise_for_status()`.

        Raises:
            httpx.HTTPError: If the request fails (connection error, timeout, or HTTP error status).
        """
        try:
            response = self._client.get(f"{self.base_url}{url}", **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            logger.error(f"GET {url} failed: {e}")
            raise

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST to `base_url` + `url`, raising on any connection/timeout/HTTP-status failure.

        Args:
            url (str): Path to post to, joined onto `base_url` (or a full URL if unset).
            **kwargs (Any): Extra kwargs forwarded to `httpx.Client.post()` (e.g. `json=`).

        Returns:
            httpx.Response: The response, after `raise_for_status()`.

        Raises:
            httpx.HTTPError: If the request fails (connection error, timeout, or HTTP error status).
        """
        try:
            response = self._client.post(f"{self.base_url}{url}", **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            logger.error(f"POST {url} failed: {e}")
            raise
