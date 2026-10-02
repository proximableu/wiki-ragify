"""Wikipedia fetcher — text and outbound-link extraction.

Wraps the Wikipedia REST API using ``requests``. Same as the reference implementation:
plain-text extracts plus paginated intra-wikipedia links (following the ``continue`` cursor).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

from ..config import Config
from ..logging_setup import get_logger
from .errors import FetchError

logger = get_logger(__name__)


class WikiPageFetcher:
    """Handles fetching text and links from Wikipedia"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.headers = {"User-Agent": config.wiki_user_agent}

    def fetch_text(self, title: str, lang: str | None = None) -> Optional[str]:
        """Fetch the plain-text extract for a title; return None on any failure."""
        lang = lang or self.config.wiki_lang
        url = "https://{lang}.wikipedia.org/w/api.php".format(lang=lang)
        params = {
            "action": "query",
            "prop": "extracts",
            "redirects": "1",
            "titles": title,
            "explaintext": "1",
            "format": "json",
        }
        try:
            r = requests.get(url, params=params, headers=self.headers, timeout=self.config.request_timeout)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            for p in pages.values():
                if "missing" in p:
                    return None
                return p.get("extract", "")
        except Exception as e:  # noqa: BLE001
            raise FetchError(f"{title}: {e}") from e

    def fetch_links(self, title: str, lang: str | None = None) -> List[str]:
        """Fetch all outgoing intra-wikipedia links for a title, following pagination."""
        lang = lang or self.config.wiki_lang
        url = "https://{lang}.wikipedia.org/w/api.php".format(lang=lang)
        params = {
            "action": "query",
            "prop": "links",
            "plnamespace": 0,
            "pllimit": "max",
            "titles": title,
            "format": "json",
        }
        out: List[str] = []
        while True:
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=self.config.request_timeout)
                r.raise_for_status()
                pages = r.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    out.extend(link["title"] for link in p.get("links", []))
                if "continue" in r.json():
                    params.update(r.json()["continue"])
                else:
                    break
            except Exception as e:  # noqa: BLE001
                raise FetchError(f"{title}: {e}") from e
        return out
