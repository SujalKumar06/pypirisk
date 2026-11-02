"""Ingest stage: crawl (or replay) every source into five tables.

Ray tasks crawl PyPI, ecosyste.ms, git repositories and (with a token) GitHub GraphQL
under shared per-host rate limits. Every response is cached on disk, which makes crawls
resumable and lets the offline ``fixture`` profile replay a recorded cache.
"""
