"""All chatbot model-facing prompt text.

Business modules may pass evidence and routing data into these builders, but
they must not define model instructions of their own.
"""

from .chat_prompts import *  # noqa: F401,F403
