"""Tests for llm_prompt_tag."""

import pytest

from llm_prompt_tag import (
    PromptTag,
    PromptTagger,
    TaggedPrompt,
    compose,
    from_dict,
    tag,
    to_dict,
)

# ---------------------------------------------------------------------------
# PromptTagger.tag()
# ---------------------------------------------------------------------------


def test_tag_creates_tagged_prompt():
    t = PromptTagger()
    tp = t.tag("hello world", "intro")
    assert isinstance(tp, TaggedPrompt)
    assert tp.text == "hello world"
    assert tp.tag.name == "intro"


def test_tag_version_none_by_default():
    t = PromptTagger()
    tp = t.tag("text", "section")
    assert tp.tag.version is None


def test_tag_version_preserved():
    t = PromptTagger()
    tp = t.tag("text", "section", version="2.1")
    assert tp.tag.version == "2.1"


def test_tag_extra_kwargs_go_to_metadata():
    t = PromptTagger()
    tp = t.tag("text", "persona", author="alice", priority=5)
    assert tp.tag.metadata["author"] == "alice"
    assert tp.tag.metadata["priority"] == 5


def test_tag_cache_hint_true():
    t = PromptTagger()
    tp = t.tag("text", "system", cache_hint=True)
    assert tp.tag.cache_hint is True


def test_tag_cache_hint_false_by_default():
    t = PromptTagger()
    tp = t.tag("text", "system")
    assert tp.tag.cache_hint is False


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------


def test_prompt_tag_is_frozen():
    pt = PromptTag(name="intro")
    with pytest.raises((AttributeError, TypeError)):
        pt.name = "other"  # type: ignore[misc]


def test_tagged_prompt_is_frozen():
    pt = PromptTag(name="intro")
    tp = TaggedPrompt(text="hello", tag=pt)
    with pytest.raises((AttributeError, TypeError)):
        tp.text = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_returns_text_unchanged():
    t = PromptTagger()
    tp = t.tag("You are a helpful assistant.", "system")
    assert t.render(tp) == "You are a helpful assistant."


def test_render_empty_text():
    t = PromptTagger()
    tp = t.tag("", "empty")
    assert t.render(tp) == ""


# ---------------------------------------------------------------------------
# render_with_markers()
# ---------------------------------------------------------------------------


def test_render_with_markers_contains_tag_name():
    t = PromptTagger()
    tp = t.tag("content", "intro")
    out = t.render_with_markers(tp)
    assert "<!-- tag:intro -->" in out


def test_render_with_markers_includes_version():
    t = PromptTagger()
    tp = t.tag("content", "intro", version="3")
    out = t.render_with_markers(tp)
    assert "v:3" in out


def test_render_with_markers_no_version_omits_v():
    t = PromptTagger()
    tp = t.tag("content", "intro")
    out = t.render_with_markers(tp)
    assert "v:" not in out


def test_render_with_markers_wraps_text():
    t = PromptTagger()
    tp = t.tag("hello", "greet")
    out = t.render_with_markers(tp)
    assert out == "<!-- tag:greet -->hello<!-- /tag:greet -->"


def test_render_with_markers_wraps_text_with_version():
    t = PromptTagger()
    tp = t.tag("hello", "greet", version="1")
    out = t.render_with_markers(tp)
    assert out == "<!-- tag:greet v:1 -->hello<!-- /tag:greet -->"


# ---------------------------------------------------------------------------
# strip_markers()
# ---------------------------------------------------------------------------


def test_strip_markers_removes_markers():
    t = PromptTagger()
    tp = t.tag("clean text", "sec")
    marked = t.render_with_markers(tp)
    stripped = t.strip_markers(marked)
    assert stripped == "clean text"


def test_strip_markers_no_markers_unchanged():
    t = PromptTagger()
    plain = "plain text here"
    assert t.strip_markers(plain) == "plain text here"


def test_strip_markers_handles_multiple_markers():
    t = PromptTagger()
    tp1 = t.tag("part one", "a")
    tp2 = t.tag("part two", "b")
    combined = t.render_with_markers(tp1) + " " + t.render_with_markers(tp2)
    stripped = t.strip_markers(combined)
    assert stripped == "part one part two"


# ---------------------------------------------------------------------------
# extract_tags()
# ---------------------------------------------------------------------------


def test_extract_tags_finds_single_section():
    t = PromptTagger()
    tp = t.tag("hello world", "intro")
    marked = t.render_with_markers(tp)
    result = t.extract_tags(marked)
    assert len(result) == 1
    assert result[0][0] == "intro"


def test_extract_tags_finds_multiple_sections():
    t = PromptTagger()
    tp1 = t.tag("section A", "alpha")
    tp2 = t.tag("section B", "beta")
    text = t.render_with_markers(tp1) + "\n" + t.render_with_markers(tp2)
    result = t.extract_tags(text)
    assert len(result) == 2


def test_extract_tags_returns_name_content_tuples():
    t = PromptTagger()
    tp = t.tag("my content", "block")
    marked = t.render_with_markers(tp)
    result = t.extract_tags(marked)
    name, content = result[0]
    assert name == "block"
    assert content == "my content"


def test_extract_tags_empty_text_returns_empty():
    t = PromptTagger()
    assert t.extract_tags("") == []


def test_extract_tags_no_markers_returns_empty():
    t = PromptTagger()
    assert t.extract_tags("just plain text") == []


def test_extract_tags_versioned_marker():
    t = PromptTagger()
    tp = t.tag("versioned content", "sec", version="2")
    marked = t.render_with_markers(tp)
    result = t.extract_tags(marked)
    assert len(result) == 1
    assert result[0] == ("sec", "versioned content")


# ---------------------------------------------------------------------------
# compose()
# ---------------------------------------------------------------------------


def test_compose_joins_with_separator():
    t = PromptTagger()
    tp1 = t.tag("first", "a")
    tp2 = t.tag("second", "b")
    out = t.compose(tp1, tp2)
    assert out == "first\n\nsecond"


def test_compose_skips_empty_text():
    t = PromptTagger()
    tp1 = t.tag("hello", "a")
    tp2 = t.tag("", "empty")
    tp3 = t.tag("world", "c")
    out = t.compose(tp1, tp2, tp3)
    assert out == "hello\n\nworld"


def test_compose_custom_separator():
    t = PromptTagger()
    tp1 = t.tag("A", "a")
    tp2 = t.tag("B", "b")
    out = t.compose(tp1, tp2, separator=" | ")
    assert out == "A | B"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def test_module_tag_function_works():
    tp = tag("text", "label", version="1", cache_hint=True, key="val")
    assert isinstance(tp, TaggedPrompt)
    assert tp.tag.name == "label"
    assert tp.tag.version == "1"
    assert tp.tag.cache_hint is True
    assert tp.tag.metadata["key"] == "val"


def test_module_compose_function_works():
    tp1 = tag("foo", "a")
    tp2 = tag("bar", "b")
    out = compose(tp1, tp2, separator="---")
    assert out == "foo---bar"


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_includes_expected_keys():
    tp = tag("some text", "sect", version="3", cache_hint=True, extra="data")
    d = to_dict(tp)
    assert d["text"] == "some text"
    assert d["name"] == "sect"
    assert d["version"] == "3"
    assert d["cache_hint"] is True
    assert d["metadata"]["extra"] == "data"


def test_from_dict_round_trip():
    original = tag("round trip", "rt", version="v2", cache_hint=False, foo="bar")
    d = to_dict(original)
    restored = from_dict(d)
    assert restored.text == original.text
    assert restored.tag.name == original.tag.name
    assert restored.tag.version == original.tag.version
    assert restored.tag.cache_hint == original.tag.cache_hint
    assert restored.tag.metadata == original.tag.metadata


def test_cache_hint_true_preserved_in_round_trip():
    tp = tag("text", "name", cache_hint=True)
    restored = from_dict(to_dict(tp))
    assert restored.tag.cache_hint is True


def test_to_dict_version_none():
    tp = tag("text", "name")
    d = to_dict(tp)
    assert d["version"] is None
