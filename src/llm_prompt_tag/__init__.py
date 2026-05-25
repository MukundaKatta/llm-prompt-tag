"""llm-prompt-tag: Tag prompt sections with named labels and metadata."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptTag:
    """Metadata label attached to a prompt section."""

    name: str
    version: str | None = None
    cache_hint: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaggedPrompt:
    """A prompt text with an associated PromptTag."""

    text: str
    tag: PromptTag


class PromptTagger:
    """Create, render, compose, and parse tagged prompt sections."""

    def tag(
        self,
        text: str,
        name: str,
        version: str | None = None,
        cache_hint: bool = False,
        **metadata: Any,
    ) -> TaggedPrompt:
        """Create a TaggedPrompt from text and tag parameters."""
        return TaggedPrompt(
            text=text,
            tag=PromptTag(name=name, version=version, cache_hint=cache_hint, metadata=metadata),
        )

    def render(self, tagged: TaggedPrompt) -> str:
        """Return the prompt text unchanged (tags are metadata-only)."""
        return tagged.text

    def render_with_markers(self, tagged: TaggedPrompt) -> str:
        """Wrap text in HTML-comment markers encoding the tag name and version.

        Format with version:    <!-- tag:name v:version -->text<!-- /tag:name -->
        Format without version: <!-- tag:name -->text<!-- /tag:name -->
        """
        name = tagged.tag.name
        version = tagged.tag.version
        if version is not None:
            open_marker = f"<!-- tag:{name} v:{version} -->"
        else:
            open_marker = f"<!-- tag:{name} -->"
        close_marker = f"<!-- /tag:{name} -->"
        return f"{open_marker}{tagged.text}{close_marker}"

    def strip_markers(self, text: str) -> str:
        """Remove all <!-- tag:... --> and <!-- /tag:... --> HTML comment markers."""
        # Remove opening markers (with or without version)
        text = re.sub(r"<!-- tag:[^>]+ -->", "", text)
        # Remove closing markers
        text = re.sub(r"<!-- /tag:[^>]+ -->", "", text)
        return text

    def extract_tags(self, text: str) -> list[tuple[str, str]]:
        """Find all marked sections and return [(tag_name, content), ...].

        Content is the text between opening and closing markers.
        Handles both versioned and unversioned opening markers.
        """
        # Match <!-- tag:NAME --> or <!-- tag:NAME v:VER --> ... <!-- /tag:NAME -->
        pattern = re.compile(
            r"<!-- tag:(\w[\w.-]*?)(?:\s+v:[^>]*)? -->(.+?)<!-- /tag:\1 -->",
            re.DOTALL,
        )
        return [(m.group(1), m.group(2)) for m in pattern.finditer(text)]

    def compose(self, *tagged: TaggedPrompt, separator: str = "\n\n") -> str:
        """Render each TaggedPrompt and join with separator, skipping empty text."""
        parts = [self.render(t) for t in tagged if t.text]
        return separator.join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_tagger = PromptTagger()


def tag(
    text: str,
    name: str,
    version: str | None = None,
    cache_hint: bool = False,
    **metadata: Any,
) -> TaggedPrompt:
    """Convenience wrapper: PromptTagger().tag(...)"""
    return _tagger.tag(text, name, version=version, cache_hint=cache_hint, **metadata)


def compose(*tagged: TaggedPrompt, separator: str = "\n\n") -> str:
    """Convenience wrapper: PromptTagger().compose(...)"""
    return _tagger.compose(*tagged, separator=separator)


def to_dict(tagged: TaggedPrompt) -> dict:
    """Serialize a TaggedPrompt to a plain dict for JSON storage."""
    return {
        "text": tagged.text,
        "name": tagged.tag.name,
        "version": tagged.tag.version,
        "cache_hint": tagged.tag.cache_hint,
        "metadata": dict(tagged.tag.metadata),
    }


def from_dict(d: dict) -> TaggedPrompt:
    """Deserialize a TaggedPrompt from a plain dict."""
    pt = PromptTag(
        name=d["name"],
        version=d.get("version"),
        cache_hint=d.get("cache_hint", False),
        metadata=d.get("metadata", {}),
    )
    return TaggedPrompt(text=d["text"], tag=pt)


__all__ = [
    "PromptTag",
    "TaggedPrompt",
    "PromptTagger",
    "tag",
    "compose",
    "to_dict",
    "from_dict",
]
