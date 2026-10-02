#!/usr/bin/env python3
"""
Wiki Crawler with LLM Gating (Structured Outputs Version)
---------------------------------------------------------
Crawls Wikipedia pages (depth=1 only), gates each page via the centralized
Ollama gateway, and saves only accepted pages.

The list of accepted titles is written verbatim from the cache (``accepted``
is keyed by the ORIGINAL Wikipedia title, not reconstructed from a filename),
so downstream phase 2 re-feeds genuine titles straight into the Wikipedia API
instead of a lossy reverse-encoding of the on-disk filename.
"""

import argparse
import json
import time
import os
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from dotenv import load_dotenv
from ollama_gateway import PipelineConfig, LLMEvaluator

# Surface .env overrides for the crawler's own infrastructure settings so the
# values in .env (WIKI_USER_AGENT, REQUEST_TIMEOUT, RATE_LIMIT_DELAY, ...) actually
# take effect instead of being silently ignored. load_dotenv() also runs as a side
# effect of importing ollama_gateway, but calling it here makes the dependency explicit.
load_dotenv()


class Color(Enum):
    RESET = "\033[0m"
    RED = "\033[1;91m"
    GREEN = "\033[1;92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def colorize(msg: str, color: Color) -> str:
    return f"{color.value}{msg}{Color.RESET.value}"


@dataclass
class CrawlerConfig:
    """Configuration specific to Wikipedia fetching and text processing."""
    wiki_api: str = "https://{lang}.wikipedia.org/w/api.php"
    user_agent: str = os.getenv(
        "WIKI_USER_AGENT", "WikiLLMGateCrawler/1.0 (mailto:proximableu@gmail.com)"
    )
    min_words: int = 40
    max_tokens: int = 2048
    # Minimum character length for a page to be worth keeping. Configurable so
    # stricter/looser bodies can be requested via .env without code edits.
    min_text_length: int = int(os.getenv("MIN_TEXT_LENGTH", "1500"))
    # Delay (s) between Wikipedia requests to avoid throttling.
    rate_limit_delay: float = float(os.getenv("RATE_LIMIT_DELAY", "0.35"))
    # Per-request timeout (s) for Wikipedia API calls.
    timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    # Enforced delay between requests; kept as a separate attribute from `timeout`
    # because it controls pacing, not socket liveness.
    title_length_limit: int = int(os.getenv("MAX_TITLE_LENGTH", "200"))


@dataclass
class Cache:
    """Cache for accepted and rejected pages, keyed by original Wikipedia title.

    ``accepted`` maps original title -> on-disk filename so the canonical list of
    accepted titles is available without reverse-engineering the (non-invertible)
    filename encoding. ``rejected`` stores titles we have already declined.
    """
    accepted: Dict[str, str] = field(default_factory=dict)
    rejected: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Cache":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    accepted=data.get("accepted", {}),
                    rejected=data.get("rejected", []),
                )
            except Exception:
                pass
        return cls()

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"accepted": self.accepted, "rejected": self.rejected},
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


class WikiPageFetcher:
    """Handles fetching text and links from Wikipedia"""

    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.headers = {"User-Agent": config.user_agent}

    def fetch_text(self, title: str, lang: str = "en") -> Optional[str]:
        url = self.config.wiki_api.format(lang=lang)
        params = {
            "action": "query", "prop": "extracts", "redirects": "1",
            "titles": title, "explaintext": "1", "format": "json"
        }
        try:
            r = requests.get(url, params=params, headers=self.headers, timeout=self.config.timeout)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            for p in pages.values():
                if "missing" in p:
                    return None
                return p.get("extract", "")
        except Exception as e:
            print(colorize(f"[FETCH_ERR] {title}: {e}", Color.RED))
            return None

    def fetch_links(self, title: str, lang: str = "en") -> List[str]:
        url = self.config.wiki_api.format(lang=lang)
        params = {
            "action": "query", "prop": "links", "plnamespace": 0,
            "pllimit": "max", "titles": title, "format": "json"
        }
        out: List[str] = []
        while True:
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=self.config.timeout)
                r.raise_for_status()
                pages = r.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    out.extend(link["title"] for link in p.get("links", []))
                if "continue" in r.json():
                    params.update(r.json()["continue"])
                else:
                    break
            except Exception as e:
                print(colorize(f"[LINK_ERR] {title}: {e}", Color.RED))
                break
        return out


class PageProcessor:
    """Handles text processing and page evaluation"""

    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config

    @staticmethod
    def title_to_filename(title: str, length_limit: int = 200) -> str:
        safe = title.replace("/", "_").replace(" ", "_").replace("&", "_")
        return safe[:length_limit] + ".txt"

    def extract_first_paragraph(self, text: str) -> Optional[str]:
        if not text:
            return None
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lead_section = text.split("==")[0].strip()
        words = lead_section.split()

        if len(words) < self.config.min_words:
            words = text.split()

        return " ".join(words[:self.config.max_tokens]) if words else None

    def is_bad_page(self, title: str, text: str) -> bool:
        if not text or "(disambiguation)" in title.lower() or title.startswith("List of"):
            return True
        return len(text) < self.config.min_text_length


class WikiCrawler:
    """Main crawler class orchestrating the entire process"""

    def __init__(self, config: CrawlerConfig, evaluator: LLMEvaluator, output_dir: Path, lang: str = "en") -> None:
        self.config = config
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.lang = lang
        self.fetcher = WikiPageFetcher(config)
        self.processor = PageProcessor(config)
        self.cache_file = output_dir / "download_cache.json"
        self.cache = Cache.load(self.cache_file)

    def process_seed_page(self, title: str, gating_prompt: str) -> Tuple[bool, str]:
        """Process a single seed page. Returns (accepted: bool, filename: str)."""
        fname = self.processor.title_to_filename(title, self.config.title_length_limit)
        if title in self.cache.accepted:
            print(colorize(f"[=SKIP=] {title} : [ACCEPT]", Color.YELLOW))
            return True, fname
        if title in self.cache.rejected:
            print(colorize(f"[=SKIP=] {title} : [REJECT]", Color.YELLOW))
            return False, fname

        text = self.fetcher.fetch_text(title, self.lang)
        if not text or self.processor.is_bad_page(title, text):
            self.cache.rejected.append(title)
            return False, fname

        para = self.processor.extract_first_paragraph(text)
        if not para:
            self.cache.rejected.append(title)
            return False, fname

        accepted = self.evaluator.evaluate(para, gating_prompt)
        if accepted:
            self.save_accepted_page(title, text, fname)
            return True, fname
        else:
            self.cache.rejected.append(title)
            print(colorize(f"[REJECT] {title}", Color.MAGENTA))
            return False, fname

    def process_linked_pages(self, seed_title: str, gating_prompt: str) -> Tuple[int, List[str]]:
        """Process all links from the seed page. Returns (accepted_count, accepted_titles)."""
        print(colorize(f"[INFO] Fetching links from seed: {seed_title}", Color.BLUE))
        raw_links = self.fetcher.fetch_links(seed_title, self.lang)
        if not raw_links:
            print(colorize(f"[WARN] No links found on {seed_title}", Color.YELLOW))
            return 0, []

        accepted_count = 0
        accepted_titles: List[str] = []
        visited: Set[str] = set()

        for link in raw_links:
            if link in visited:
                continue
            visited.add(link)
            fname = self.processor.title_to_filename(link, self.config.title_length_limit)

            if link in self.cache.accepted:
                print(colorize(f"[=SKIP=] {link} : [ACCEPT]", Color.YELLOW))
                accepted_count += 1
                accepted_titles.append(link)
                continue
            if link in self.cache.rejected:
                print(colorize(f"[=SKIP=] {link} : [REJECT]", Color.YELLOW))
                continue

            text = self.fetcher.fetch_text(link, self.lang)
            if not text or self.processor.is_bad_page(link, text):
                self.cache.rejected.append(link)
                continue

            para = self.processor.extract_first_paragraph(text)
            if not para:
                self.cache.rejected.append(link)
                continue

            accepted = self.evaluator.evaluate(para, gating_prompt)
            if accepted:
                self.save_accepted_page(link, text, fname)
                accepted_count += 1
                accepted_titles.append(link)
            else:
                self.cache.rejected.append(link)
                print(colorize(f"[REJECT] {link}", Color.MAGENTA))

            self.cache.save(self.cache_file)
            # Enforce strict 1 request/second pacing to avoid Wikipedia API throttling
            time.sleep(self.config.rate_limit_delay)

        return accepted_count, accepted_titles

    def save_accepted_page(self, title: str, text: str, fname: str) -> None:
        path = self.output_dir / fname
        if path.exists():
            print(colorize(f"[=SKIP=] Already saved: {fname}", Color.YELLOW))
            return
        path.write_text(text, encoding="utf-8")
        self.cache.accepted[title] = fname
        print(colorize(f"[ACCEPT] {title}", Color.CYAN))

    def crawl(self, start_title: str, gating_prompt: str) -> Tuple[int, List[str]]:
        """Crawl one seed page + its direct links. Returns (accepted_count, accepted_titles)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        accepted_count = 0
        accepted_titles: List[str] = []

        seed_success, _ = self.process_seed_page(start_title, gating_prompt)
        if seed_success:
            accepted_count += 1
            accepted_titles.append(start_title)

        new_accepted_count, new_accepted_titles = self.process_linked_pages(start_title, gating_prompt)
        accepted_count += new_accepted_count
        accepted_titles.extend(new_accepted_titles)

        self.cache.save(self.cache_file)
        print(colorize(f"[DONE] Seed '{start_title}': {accepted_count} accepted", Color.CYAN))
        return accepted_count, accepted_titles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wiki Crawler with LLM Gating (depth=1 only)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--start-page", type=str, help="Single Wikipedia title or URL")
    group.add_argument("--start-list", type=str, help="Path to file with one title per line")
    p.add_argument("gating_prompt_path", type=Path, help="Path to gating prompt file")
    p.add_argument("output_dir", type=Path, help="Directory to save accepted pages")
    p.add_argument("--lang", default="en", help="Wikipedia language code")
    return p.parse_args()


def collect_start_titles(args: argparse.Namespace) -> List[str]:
    """Resolve the list of seed titles from either --start-page or --start-list."""
    if args.start_page:
        return [line.strip() for line in args.start_page.splitlines() if line.strip()]
    # --start-list: one title per line, blank lines ignored
    return [line.strip() for line in Path(args.start_list).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if not args.gating_prompt_path.exists():
        raise FileNotFoundError(f"Gating prompt not found: {args.gating_prompt_path}")

    gating_prompt = args.gating_prompt_path.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize centralized configuration and evaluator
    pipeline_config = PipelineConfig()
    evaluator = LLMEvaluator(pipeline_config)
    crawler_config = CrawlerConfig()
    crawler = WikiCrawler(crawler_config, evaluator, args.output_dir, args.lang)

    titles = collect_start_titles(args)
    if not titles:
        print(colorize("[WARN] No seed titles provided.", Color.YELLOW))
        evaluator.close()
        return

    try:
        for i, title in enumerate(titles, 1):
            print(colorize(f"\n=== Starting seed {i}/{len(titles)}: {title} ===", Color.CYAN))
            crawler.crawl(title, gating_prompt)

        # accepted_pages.list now holds the ORIGINAL titles from the cache (no filename round-trip),
        # so downstream phase 2 re-feeds genuine titles straight into the Wikipedia API.
        if crawler.cache.accepted:
            list_content = "\n".join(crawler.cache.accepted.keys())
            list_path = args.output_dir / "accepted_pages.list"
            list_path.write_text(list_content + "\n", encoding="utf-8")
            print(colorize(f"[LIST] Saved {len(crawler.cache.accepted)} titles to {list_path}", Color.GREEN))
        else:
            print(colorize("[WARN] No pages accepted; no list generated.", Color.YELLOW))

        print(colorize(f"\n[FINAL] Crawl complete. Cache saved to {crawler.cache_file}", Color.GREEN))
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
