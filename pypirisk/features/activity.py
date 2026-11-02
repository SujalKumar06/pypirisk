"""Identify automation accounts in commit and issue authors."""


def is_bot_author(author: str, bot_markers: list[str]) -> bool:
    """Tell whether an author is an automation account.

    Args:
        author: Login, name or email.
        bot_markers: Case-insensitive substrings that identify bots.

    Returns:
        ``True`` if any marker occurs in ``author``.
    """
    author = author.lower()
    return any(marker.lower() in author for marker in bot_markers)
