# Prompt catalog

`chat_prompts.py` is the canonical location for every model-facing instruction
owned by the chatbot application. It contains:

- persona system-prompt composition;
- light and creative route instructions;
- private-memory boundaries, audit, and rewrite prompts;
- inline memory-candidate and answer-consolidation prompts;
- visual evidence prompts;
- prompt-only context formatting used by the chat and memory calls.

`persona.toml` now contains only character data and user-facing fallback text.
The associative-memory engine owns its separate extraction, retrieval, audit,
and graph-growth catalog under its own `config/prompt_config` directory.
