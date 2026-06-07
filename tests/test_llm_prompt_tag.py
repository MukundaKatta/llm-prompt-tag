"""Standard-library ``unittest`` suite for llm-prompt-tag.

Run with::

    python3 -m unittest discover -s tests

The suite deliberately uses only the standard library (no pytest) so it runs
in any clean Python environment without installing third-party test deps.
"""

import os
import sys
import unittest

# Make the ``src`` layout importable without an editable install so the suite
# runs standalone via ``python3 -m unittest discover -s tests``.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_prompt_tag import PromptSection, TaggedPrompt  # noqa: E402


class TaggedPromptBasicsTest(unittest.TestCase):
    def test_add_and_render(self):
        p = TaggedPrompt()
        p.add("persona", "You are helpful.")
        p.add("task", "Answer briefly.")
        text = p.render()
        self.assertIn("You are helpful.", text)
        self.assertIn("Answer briefly.", text)

    def test_default_separator(self):
        p = TaggedPrompt()
        p.add("a", "First.")
        p.add("b", "Second.")
        self.assertEqual(p.render(), "First.\n\nSecond.")

    def test_custom_separator(self):
        p = TaggedPrompt(separator="\n---\n")
        p.add("a", "First.")
        p.add("b", "Second.")
        self.assertIn("---", p.render())

    def test_priority_order_descending(self):
        p = TaggedPrompt()
        p.add("low", "low priority text", priority=1)
        p.add("high", "high priority text", priority=10)
        text = p.render()
        self.assertLess(text.index("high priority"), text.index("low priority"))

    def test_add_is_chainable(self):
        p = TaggedPrompt()
        result = p.add("a", "text").add("b", "text")
        self.assertIs(result, p)
        self.assertEqual(len(p), 2)

    def test_add_same_label_overwrites(self):
        p = TaggedPrompt()
        p.add("s", "first")
        p.add("s", "second")
        self.assertEqual(len(p), 1)
        self.assertEqual(p.get("s").content, "second")


class EnableDisableTest(unittest.TestCase):
    def test_disable_section_hides_it(self):
        p = TaggedPrompt()
        p.add("visible", "Show me.")
        p.add("hidden", "Hide me.")
        p.disable("hidden")
        text = p.render()
        self.assertIn("Show me", text)
        self.assertNotIn("Hide me", text)

    def test_enable_section_shows_it(self):
        p = TaggedPrompt()
        p.add("s", "Content.", enabled=False)
        self.assertNotIn("Content.", p.render())
        p.enable("s")
        self.assertIn("Content.", p.render())

    def test_enable_disable_unknown_label_is_noop(self):
        p = TaggedPrompt()
        # Should not raise even though the label does not exist.
        p.enable("missing")
        p.disable("missing")
        self.assertEqual(len(p), 0)


class MutationTest(unittest.TestCase):
    def test_update_existing_section(self):
        p = TaggedPrompt()
        p.add("s", "old text")
        p.update("s", "new text")
        self.assertEqual(p.get("s").content, "new text")

    def test_update_creates_missing_section(self):
        p = TaggedPrompt()
        p.update("new", "fresh content")
        self.assertIn("new", p)
        self.assertEqual(p.get("new").content, "fresh content")

    def test_remove_section(self):
        p = TaggedPrompt()
        p.add("s", "text")
        p.remove("s")
        self.assertNotIn("s", p)

    def test_remove_unknown_label_is_noop(self):
        p = TaggedPrompt()
        p.remove("missing")  # should not raise
        self.assertEqual(len(p), 0)


class VariableSubstitutionTest(unittest.TestCase):
    def test_full_substitution(self):
        p = TaggedPrompt()
        p.add("ctx", "Working in {language}.")
        self.assertEqual(p.render(variables={"language": "Python"}), "Working in Python.")

    def test_missing_variable_preserves_placeholder(self):
        # A missing variable must not crash and must not blank the section.
        p = TaggedPrompt()
        p.add("s", "Hello {name}.")
        self.assertEqual(p.render(variables={}), "Hello {name}.")

    def test_partial_substitution_keeps_unknown_placeholders(self):
        # Regression: previously a single missing key discarded ALL
        # substitution for the section.
        p = TaggedPrompt()
        p.add("s", "Hi {name}, you use {language}.")
        self.assertEqual(
            p.render(variables={"name": "Alice"}),
            "Hi Alice, you use {language}.",
        )

    def test_literal_braces_do_not_break_substitution(self):
        # Regression: literal braces (e.g. a JSON example) previously caused
        # the whole section to fall back to unsubstituted text, dropping
        # legitimate variables.
        p = TaggedPrompt()
        p.add("s", 'Return JSON like {"k": 1} for {name}.')
        self.assertEqual(
            p.render(variables={"name": "Bob"}),
            'Return JSON like {"k": 1} for {name}.',
        )

    def test_no_variables_argument_leaves_braces_untouched(self):
        p = TaggedPrompt()
        p.add("s", "Keep {raw} as-is.")
        self.assertEqual(p.render(), "Keep {raw} as-is.")

    def test_unmatched_brace_falls_back_to_raw(self):
        p = TaggedPrompt()
        p.add("s", "Unmatched { brace and {name}")
        # Malformed template: do not raise, return content unchanged.
        self.assertEqual(
            p.render(variables={"name": "X"}),
            "Unmatched { brace and {name}",
        )


class RenderingFormatsTest(unittest.TestCase):
    def test_render_with_markers(self):
        p = TaggedPrompt()
        p.add("persona", "You are helpful.")
        text = p.render_with_markers()
        self.assertIn("<persona>", text)
        self.assertIn("</persona>", text)
        self.assertIn("You are helpful.", text)

    def test_render_with_markers_applies_variables(self):
        p = TaggedPrompt()
        p.add("ctx", "Working in {language}.")
        text = p.render_with_markers(variables={"language": "Rust"})
        self.assertIn("<ctx>", text)
        self.assertIn("Working in Rust.", text)

    def test_as_message_default_role(self):
        p = TaggedPrompt()
        p.add("s", "Content.")
        msg = p.as_message()
        self.assertEqual(msg["role"], "system")
        self.assertIn("Content.", msg["content"])

    def test_as_message_custom_role(self):
        p = TaggedPrompt()
        p.add("s", "text")
        self.assertEqual(p.as_message(role="user")["role"], "user")

    def test_as_message_applies_variables(self):
        p = TaggedPrompt()
        p.add("ctx", "Hello, {name}!")
        self.assertEqual(p.as_message(variables={"name": "Bob"})["content"], "Hello, Bob!")


class SerializationTest(unittest.TestCase):
    def test_to_dict_and_from_dict_roundtrip(self):
        p = TaggedPrompt()
        p.add("persona", "You are helpful.")
        p.add("task", "Be brief.", priority=5)
        restored = TaggedPrompt.from_dict(p.to_dict())
        self.assertIn("persona", restored)
        self.assertIn("task", restored)
        self.assertEqual(restored.get("task").priority, 5)

    def test_roundtrip_preserves_separator_and_metadata(self):
        p = TaggedPrompt(separator="\n###\n")
        p.add("persona", "You are helpful.", priority=2, source="manual")
        p.add("task", "Be brief.", priority=1)
        restored = TaggedPrompt.from_dict(p.to_dict())
        self.assertEqual(restored.get("persona").metadata, {"source": "manual"})
        self.assertIn("###", restored.render())

    def test_clone_is_independent(self):
        p = TaggedPrompt()
        p.add("s", "original")
        clone = p.clone()
        p.update("s", "changed")
        self.assertEqual(clone.get("s").content, "original")


class IntrospectionTest(unittest.TestCase):
    def test_len(self):
        p = TaggedPrompt()
        p.add("a", "text").add("b", "text")
        self.assertEqual(len(p), 2)

    def test_contains(self):
        p = TaggedPrompt()
        p.add("s", "text")
        self.assertIn("s", p)
        self.assertNotIn("other", p)

    def test_labels(self):
        p = TaggedPrompt()
        p.add("x", "text").add("y", "text")
        self.assertEqual(set(p.labels()), {"x", "y"})

    def test_get_missing_returns_none(self):
        self.assertIsNone(TaggedPrompt().get("absent"))


class PromptSectionTest(unittest.TestCase):
    def test_to_dict_from_dict(self):
        s = PromptSection(label="test", content="hello", priority=5)
        restored = PromptSection.from_dict(s.to_dict())
        self.assertEqual(restored.label, "test")
        self.assertEqual(restored.priority, 5)

    def test_from_dict_defaults(self):
        restored = PromptSection.from_dict({"label": "l", "content": "c"})
        self.assertTrue(restored.enabled)
        self.assertEqual(restored.priority, 0)
        self.assertEqual(restored.metadata, {})

    def test_render_with_markers(self):
        s = PromptSection(label="ctx", content="text here")
        rendered = s.render_with_markers()
        self.assertIn("<ctx>", rendered)
        self.assertIn("text here", rendered)

    def test_render_section_substitutes_variables(self):
        s = PromptSection(label="ctx", content="Use {tool}.")
        self.assertEqual(s.render(variables={"tool": "ripgrep"}), "Use ripgrep.")


if __name__ == "__main__":
    unittest.main()
