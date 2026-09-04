"""Transport-neutral message formatting helpers."""


def split_message(text: str, limit: int) -> list[str]:
    """Split a message while preferring paragraph and word boundaries."""

    if limit < 1:
        raise ValueError("message limit must be positive")
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        boundary = remaining.rfind("\n", 0, limit + 1)
        if boundary < limit // 2:
            boundary = remaining.rfind(" ", 0, limit + 1)
        if boundary < limit // 2:
            boundary = limit
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip("\n ")
    return chunks
