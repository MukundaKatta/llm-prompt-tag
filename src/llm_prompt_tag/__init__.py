"""
llm-prompt-tag: Tag prompt sections with labels and metadata; render with markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Optional

__version__ = "0.1.0"


class _SafeFormatter(Formatter):
    """A ``string.Formatter`` that tolerates missing keys.

    Unlike ``str.format_map``, this substitutes every placeholder it *can*
    resolve and leaves placeholders whose key is missing untouched (rendered
    back as ``{key}``). This means a single missing variable no longer discards
    substitution for the whole string, and it never raises ``KeyError`` /
    ``IndexError`` for unknown fields.
    """

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        if isinstance(key, str):
            if key in kwargs:
                return kwargs[key]
            # Preserve the original placeholder for unknown keys.
            return "{" + key + "}"
        return super().get_value(key, args, kwargs)


_SAFE_FORMATTER = _SafeFormatter()


def _safe_format(text: str, variables: dict[str, Any]) -> str:
    """Substitute ``variables`` into ``text`` without failing on missing keys.

    Falls back to returning ``text`` unchanged if the string contains
    malformed format syntax (e.g. an unmatched ``{``) so that a section is
    never lost just because its content was not intended as a template.
    """
    try:
        return _SAFE_FORMATTER.vformat(text, (), variables)
    except (ValueError, IndexError, KeyError):
        return text


@dataclass
class PromptSection:
    label: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0

    def render(self, variables: Optional[dict[str, Any]] = None) -> str:
        text = self.content
        if variables:
            text = _safe_format(text, variables)
        return text

    def render_with_markers(self, variables: Optional[dict[str, Any]] = None) -> str:
        text = self.render(variables)
        return f"<{self.label}>\n{text}\n</{self.label}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "content": self.content,
            "metadata": self.metadata,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptSection":
        return cls(
            label=d["label"],
            content=d["content"],
            metadata=d.get("metadata", {}),
            enabled=d.get("enabled", True),
            priority=d.get("priority", 0),
        )


class TaggedPrompt:
    """
    Compose a prompt from labeled, tagged sections.

    Usage::

        p = TaggedPrompt()
        p.add("persona", "You are a helpful coding assistant.")
        p.add("context", "The user is working on {language}.")
        p.add("instructions", "Answer briefly.", priority=10)

        text = p.render(variables={"language": "Python"})
        # returns sections joined by separator, sorted by priority desc
    """

    def __init__(self, separator: str = "\n\n") -> None:
        self._sections: dict[str, PromptSection] = {}
        self._separator = separator

    def add(
        self,
        label: str,
        content: str,
        priority: int = 0,
        enabled: bool = True,
        **metadata: Any,
    ) -> "TaggedPrompt":
        self._sections[label] = PromptSection(
            label=label, content=content, metadata=metadata, enabled=enabled, priority=priority
        )
        return self

    def update(self, label: str, content: str) -> "TaggedPrompt":
        if label in self._sections:
            self._sections[label].content = content
        else:
            self.add(label, content)
        return self

    def enable(self, label: str) -> "TaggedPrompt":
        if label in self._sections:
            self._sections[label].enabled = True
        return self

    def disable(self, label: str) -> "TaggedPrompt":
        if label in self._sections:
            self._sections[label].enabled = False
        return self

    def remove(self, label: str) -> "TaggedPrompt":
        self._sections.pop(label, None)
        return self

    def get(self, label: str) -> Optional[PromptSection]:
        return self._sections.get(label)

    def labels(self) -> list[str]:
        return list(self._sections.keys())

    def _active_sorted(self) -> list[PromptSection]:
        active = [s for s in self._sections.values() if s.enabled]
        return sorted(active, key=lambda s: s.priority, reverse=True)

    def render(self, variables: Optional[dict[str, Any]] = None) -> str:
        parts = [s.render(variables) for s in self._active_sorted()]
        return self._separator.join(parts)

    def render_with_markers(self, variables: Optional[dict[str, Any]] = None) -> str:
        parts = [s.render_with_markers(variables) for s in self._active_sorted()]
        return self._separator.join(parts)

    def as_message(
        self,
        role: str = "system",
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {"role": role, "content": self.render(variables)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [s.to_dict() for s in self._sections.values()],
            "separator": self._separator,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaggedPrompt":
        p = cls(separator=d.get("separator", "\n\n"))
        for sec in d.get("sections", []):
            s = PromptSection.from_dict(sec)
            p._sections[s.label] = s
        return p

    def clone(self) -> "TaggedPrompt":
        return TaggedPrompt.from_dict(self.to_dict())

    def __len__(self) -> int:
        return len(self._sections)

    def __contains__(self, label: str) -> bool:
        return label in self._sections


__all__ = ["TaggedPrompt", "PromptSection"]
