# llm-prompt-tag

Named, prioritized, togglable sections for composing LLM system prompts.

## Install

```
pip install llm-prompt-tag
```

## Usage

```python
from llm_prompt_tag import TaggedPrompt

p = TaggedPrompt()
p.add("persona", "You are a helpful assistant.", priority=10)
p.add("guidelines", "Always be concise.", priority=5)
p.add("context", "User is a developer.", priority=3)

print(p.render())
# Sections rendered highest-priority first

# Disable a section
p.disable("context")

# Variable substitution
p.add("greeting", "Hello, {name}!")
print(p.render(variables={"name": "Alice"}))

# Export as chat message
msg = p.as_message(role="system")
```
