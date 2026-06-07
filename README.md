# llm-prompt-tag

Named, prioritized, togglable sections for composing LLM system prompts.

`llm-prompt-tag` lets you build a system prompt out of small, labeled pieces
(persona, guidelines, context, output format, ...) instead of one giant
f-string. Each section has a **label**, a **priority** that controls ordering,
an **enabled** flag so you can switch parts on and off, and optional
**metadata**. Sections support safe `{variable}` substitution and can be
rendered as plain text, wrapped in XML-style markers, or emitted as a chat
message dict.

## Why

Production prompts grow into unwieldy multi-hundred-line strings that are hard
to reorder, A/B test, or conditionally include. This library keeps each concern
as an independent, named unit:

- **Reorder** by priority instead of cut-and-paste.
- **Toggle** sections per request (e.g. only include `few_shot` for hard tasks).
- **Substitute** variables without a single missing key blanking the section.
- **Serialize** a whole prompt to a dict (and back) for storage or diffing.

## Install

```bash
pip install llm-prompt-tag
```

The package ships type hints and a `py.typed` marker, so it works with `mypy`
and other type checkers out of the box. Requires Python 3.9+ and has **no
runtime dependencies**.

## Usage

```python
from llm_prompt_tag import TaggedPrompt

p = TaggedPrompt()
p.add("persona", "You are a helpful assistant.", priority=10)
p.add("guidelines", "Always be concise.", priority=5)
p.add("context", "User is a developer working in {language}.", priority=3)

# Sections render highest-priority first, joined by a blank line.
print(p.render(variables={"language": "Python"}))
# You are a helpful assistant.
#
# Always be concise.
#
# User is a developer working in Python.

# Toggle a section off for this render.
p.disable("context")
print("context" in p.render())  # -> False

# Wrap each section in XML-style markers (handy for Claude-style prompts).
print(p.render_with_markers())
# <persona>
# You are a helpful assistant.
# </persona>
#
# <guidelines>
# Always be concise.
# </guidelines>

# Emit a chat message dict ready for an LLM client.
msg = p.as_message(role="system", variables={"language": "Python"})
# {"role": "system", "content": "..."}
```

### Safe variable substitution

Substitution never crashes and never silently drops your text:

```python
p = TaggedPrompt()
p.add("task", 'Return JSON like {"ok": true} for user {name}.')

# Literal braces (e.g. a JSON example) are preserved, and {name} is still filled.
p.render(variables={"name": "Ada"})
# 'Return JSON like {"ok": true} for user Ada.'

# A missing variable leaves its placeholder intact instead of blanking
# the whole section.
p2 = TaggedPrompt()
p2.add("ctx", "Hi {name}, you use {tool}.")
p2.render(variables={"name": "Ada"})
# 'Hi Ada, you use {tool}.'
```

### Serialize and clone

```python
data = p.to_dict()          # plain dict, JSON-serializable
restored = TaggedPrompt.from_dict(data)
copy = p.clone()            # deep, independent copy
```

## API

### `TaggedPrompt(separator="\n\n")`

A container of labeled sections.

| Method | Description |
| --- | --- |
| `add(label, content, priority=0, enabled=True, **metadata)` | Add or replace a section. Returns `self` (chainable). |
| `update(label, content)` | Replace a section's content; creates it if missing. Returns `self`. |
| `enable(label)` / `disable(label)` | Toggle a section's `enabled` flag. No-op if the label is unknown. Returns `self`. |
| `remove(label)` | Remove a section. No-op if missing. Returns `self`. |
| `get(label)` | Return the `PromptSection`, or `None` if absent. |
| `labels()` | List of section labels in insertion order. |
| `render(variables=None)` | Plain-text render of enabled sections, highest priority first. |
| `render_with_markers(variables=None)` | Same, but each section is wrapped in `<label>...</label>`. |
| `as_message(role="system", variables=None)` | `{"role": role, "content": render(...)}`. |
| `to_dict()` / `from_dict(d)` | JSON-serializable round-trip. |
| `clone()` | Independent deep copy. |
| `len(p)`, `label in p` | Number of sections / membership test. |

### `PromptSection`

Dataclass with fields `label`, `content`, `metadata`, `enabled`, `priority`,
and methods `render(variables=None)`, `render_with_markers(variables=None)`,
`to_dict()`, and `from_dict(d)`.

## Development

Run the test suite with the standard library only (no extra installs needed):

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
