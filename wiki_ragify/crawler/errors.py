"""Crawler error hierarchy — typed so the runner can distinguish fetch vs. gate vs. split."""

from __future__ import annotations


class CrawlerError(Exception):
    """Base class for all pipeline-stage errors."""


class FetchError(CrawlerError):
    """Wikipedia request failed (network, HTTP, parse)."""


class GateError(CrawlerError):
    """LLM gate failed in an unrecoverable way (distinct from a plain rejection)."""


class SplitError(CrawlerError):
    """Text failed to split/chunk."""


class IngestionError(CrawlerError):
    """Embedding/dedup/DB write failed."""
