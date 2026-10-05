"""Crawler runner — the in-process engine for Gate 1 (article-level gating).

Emits events the UI consumes. Runs on a worker thread so the Textual event loop stays
responsive during multi-minute crawls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List

from ..pipeline.events import ProgressEvent
from ..logging_setup import get_logger
from .cache import Cache
from .errors import FetchError, GateError
from .processor import PageProcessor

logger = get_logger(__name__)


class Crawler:
    """Crawls Wikipedia from one or more seed titles, applying article-level gating."""

    def __init__(self, config, evaluator, on_event: Callable[[ProgressEvent], None] | None = None, fetcher=None):
        self.config = config
        self.evaluator = evaluator
        self.on_event = on_event
        # Optional fetcher override so the runner can be exercised offline with a
        # fake (no network). Defaults to the real Wikipedia fetcher.
        self.processor = PageProcessor(config)
        # An explicit fetcher override (e.g. a test fake); ``None`` means "build the real
        # one lazily on first use".
        self.__fetcher = fetcher
        # Crawler artifacts (accepted pages + cache) live in ``output/``, matching the
        # reference crawler and so the runner's splitter / phase-2 artifact check see them.
        self.crawl_dir = config.output_dir / config.output_dirname
        self.cache_file = self.crawl_dir / "download_cache.json"
        self.cache = Cache.load(self.cache_file)

    @property
    def fetcher(self):
        # Lazily build the real Wikipedia fetcher only on first use, so importing this
        # module needs no ``requests`` on the path.
        if self.__fetcher is None:
            from .fetcher import WikiPageFetcher

            self.__fetcher = WikiPageFetcher(self.config)
        return self.__fetcher

    def _emit(self, **kwargs):
        if self.on_event:
            self.on_event(ProgressEvent(**kwargs))

    def _save_accepted_page(self, title, text, fname):
        path = self.crawl_dir / fname
        if path.exists():
            self._emit(stage="crawl", kind="skip", message=f"[SKIP] Already saved: {fname}")
            return
        path.write_text(text, encoding="utf-8")
        self.cache.add_accepted(title, fname)
        self._emit(stage="crawl", kind="accept", message=f"[ACCEPT] {title}")

    def _process_single(self, title, gating_prompt):
        fname = self.processor.title_to_filename(title, self.config.title_length_limit)
        if title in self.cache.accepted:
            self._emit(stage="crawl", kind="skip", message=f"[=SKIP=] {title} : [ACCEPT]")
            return True, fname
        if title in self.cache.rejected:
            self._emit(stage="crawl", kind="skip", message=f"[=SKIP=] {title} : [REJECT]")
            return False, fname

        text = self.fetcher.fetch_text(title)
        if not text or self.processor.is_bad_page(title, text):
            self.cache.add_rejected(title)
            return False, fname

        para = self.processor.extract_first_paragraph(text)
        if not para:
            self.cache.add_rejected(title)
            return False, fname

        accepted = self.evaluator.evaluate(para, gating_prompt)
        if accepted:
            self._save_accepted_page(title, text, fname)
            return True, fname
        self.cache.add_rejected(title)
        self._emit(stage="crawl", kind="reject", message=f"[REJECT] {title}")
        return False, fname

    def _process_linked(self, seed_title, gating_prompt):
        self._emit(stage="crawl", kind="info", message=f"[INFO] Fetching links from seed: {seed_title}")
        raw_links = self.fetcher.fetch_links(seed_title)
        if not raw_links:
            self._emit(stage="crawl", kind="warn", message=f"[WARN] No links found on {seed_title}")
            return 0, []

        accepted_count = 0
        accepted_titles: List[str] = []
        visited = set()

        for link in raw_links:
            if link in visited:
                continue
            visited.add(link)
            fname = self.processor.title_to_filename(link, self.config.title_length_limit)

            if link in self.cache.accepted:
                self._emit(stage="crawl", kind="skip", message=f"[=SKIP=] {link} : [ACCEPT]")
                accepted_count += 1
                accepted_titles.append(link)
                continue
            if link in self.cache.rejected:
                self._emit(stage="crawl", kind="skip", message=f"[=SKIP=] {link} : [REJECT]")
                continue

            text = self.fetcher.fetch_text(link)
            if not text or self.processor.is_bad_page(link, text):
                self.cache.add_rejected(link)
                continue
            para = self.processor.extract_first_paragraph(text)
            if not para:
                self.cache.add_rejected(link)
                continue

            accepted = self.evaluator.evaluate(para, gating_prompt)
            if accepted:
                self._save_accepted_page(link, text, fname)
                accepted_count += 1
                accepted_titles.append(link)
            else:
                self.cache.add_rejected(link)
                self._emit(stage="crawl", kind="reject", message=f"[REJECT] {link}")

            self.cache.save(self.cache_file)
            self._emit(stage="crawl", kind="tick", message=f"[LINK] {link}")
            time.sleep(self.config.rate_limit_delay)

        return accepted_count, accepted_titles

    def crawl(self, start_titles: List[str], gating_prompt: str):
        """Crawl all seed pages (each + its direct links). Returns accepted titles."""
        total = len(start_titles)
        self.crawl_dir.mkdir(parents=True, exist_ok=True)
        accepted: List[str] = []

        # Seed the progress bar at 0/total *before* processing any seed, so the
        # banner opens at 0% instead of jumping straight to 100% on the first tick.
        self._emit(
            stage="crawl",
            kind="tick",
            index=0,
            total=total,
            message=f"[INFO] Starting crawl of {total} seed title(s)",
        )

        for i, title in enumerate(start_titles, 1):
            ok, _ = self._process_single(title, gating_prompt)
            _, linked = self._process_linked(title, gating_prompt)
            if ok:
                accepted.append(title)
            accepted.extend(linked)
            self.cache.save(self.cache_file)
            # Advance the bar *after* the seed (and its links) are processed, so the
            # index tracks how many of the total seeds have completed.
            self._emit(
                stage="crawl",
                kind="tick",
                index=i,
                total=total,
                message=f"=== {i}/{total}: {title} processed ===",
            )

        accepted_list = self.config.output_dir / self.config.output_dirname / "accepted_pages.list"
        accepted_list.parent.mkdir(parents=True, exist_ok=True)
        accepted_list.write_text("\n".join(self.cache.titles), encoding="utf-8")

        self._emit(stage="crawl", kind="stage_done",
                   message=f"[DONE] {len(self.cache.accepted)} accepted, {len(self.cache.rejected)} rejected")
        return self.cache.titles
