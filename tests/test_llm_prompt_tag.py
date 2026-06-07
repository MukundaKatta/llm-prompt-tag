"""Tests for llm-prompt-tag."""

from llm_prompt_tag import PromptSection, TaggedPrompt


def test_add_and_render():
    p = TaggedPrompt()
    p.add("persona", "You are helpful.")
    p.add("task", "Answer briefly.")
    text = p.render()
    assert "You are helpful." in text
    assert "Answer briefly." in text


def test_sections_separated():
    p = TaggedPrompt(separator="\n---\n")
    p.add("a", "First.")
    p.add("b", "Second.")
    text = p.render()
    assert "---" in text


def test_priority_order():
    p = TaggedPrompt()
    p.add("low", "low priority text", priority=1)
    p.add("high", "high priority text", priority=10)
    text = p.render()
    assert text.index("high priority") < text.index("low priority")


def test_disable_section():
    p = TaggedPrompt()
    p.add("visible", "Show me.")
    p.add("hidden", "Hide me.")
    p.disable("hidden")
    text = p.render()
    assert "Show me" in text
    assert "Hide me" not in text


def test_enable_section():
    p = TaggedPrompt()
    p.add("s", "Content.", enabled=False)
    p.enable("s")
    text = p.render()
    assert "Content." in text


def test_update_section():
    p = TaggedPrompt()
    p.add("s", "old text")
    p.update("s", "new text")
    assert p.get("s").content == "new text"


def test_remove_section():
    p = TaggedPrompt()
    p.add("s", "text")
    p.remove("s")
    assert "s" not in p


def test_variables_interpolation():
    p = TaggedPrompt()
    p.add("ctx", "Working in {language}.")
    text = p.render(variables={"language": "Python"})
    assert "Python" in text


def test_render_with_markers():
    p = TaggedPrompt()
    p.add("persona", "You are helpful.")
    text = p.render_with_markers()
    assert "<persona>" in text
    assert "</persona>" in text


def test_as_message():
    p = TaggedPrompt()
    p.add("s", "Content.")
    msg = p.as_message()
    assert msg["role"] == "system"
    assert "Content." in msg["content"]


def test_as_message_custom_role():
    p = TaggedPrompt()
    p.add("s", "text")
    msg = p.as_message(role="user")
    assert msg["role"] == "user"


def test_to_dict_and_from_dict():
    p = TaggedPrompt()
    p.add("persona", "You are helpful.")
    p.add("task", "Be brief.", priority=5)
    d = p.to_dict()
    p2 = TaggedPrompt.from_dict(d)
    assert "persona" in p2
    assert "task" in p2
    assert p2.get("task").priority == 5


def test_clone():
    p = TaggedPrompt()
    p.add("s", "original")
    p2 = p.clone()
    p.update("s", "changed")
    assert p2.get("s").content == "original"


def test_len():
    p = TaggedPrompt()
    p.add("a", "text").add("b", "text")
    assert len(p) == 2


def test_contains():
    p = TaggedPrompt()
    p.add("s", "text")
    assert "s" in p
    assert "other" not in p


def test_labels():
    p = TaggedPrompt()
    p.add("x", "text").add("y", "text")
    assert "x" in p.labels()
    assert "y" in p.labels()


def test_section_to_dict_from_dict():
    s = PromptSection(label="test", content="hello", priority=5)
    d = s.to_dict()
    s2 = PromptSection.from_dict(d)
    assert s2.label == "test"
    assert s2.priority == 5


def test_section_render_with_markers():
    s = PromptSection(label="ctx", content="text here")
    rendered = s.render_with_markers()
    assert "<ctx>" in rendered
    assert "text here" in rendered


def test_missing_variable_passthrough():
    p = TaggedPrompt()
    p.add("s", "Hello {name}.")
    # missing variable should not crash
    text = p.render(variables={})
    assert "Hello" in text


def test_render_with_markers_variables():
    p = TaggedPrompt()
    p.add("ctx", "Working in {language}.")
    text = p.render_with_markers(variables={"language": "Rust"})
    assert "<ctx>" in text
    assert "Working in Rust." in text


def test_as_message_variables():
    p = TaggedPrompt()
    p.add("ctx", "Hello, {name}!")
    msg = p.as_message(variables={"name": "Bob"})
    assert msg["content"] == "Hello, Bob!"


def test_update_creates_missing_section():
    p = TaggedPrompt()
    p.update("new", "fresh content")
    assert "new" in p
    assert p.get("new").content == "fresh content"


def test_to_dict_from_dict_preserves_separator_and_metadata():
    p = TaggedPrompt(separator="\n###\n")
    p.add("persona", "You are helpful.", priority=2, source="manual")
    p.add("task", "Be brief.", priority=1)
    restored = TaggedPrompt.from_dict(p.to_dict())
    assert restored._separator == "\n###\n"
    section = restored.get("persona")
    assert section.metadata == {"source": "manual"}
    # separator joins the two sections in the rendered output
    assert "###" in restored.render()


def test_get_missing_returns_none():
    p = TaggedPrompt()
    assert p.get("absent") is None
