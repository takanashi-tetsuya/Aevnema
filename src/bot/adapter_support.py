"""Public adapter integration API.

Transport implementations import from this facade.  The implementation is
separated into access control, text utilities, process runtime and dispatch.
"""

from .access import (
    require_access_policy,
    whitelist_enabled,
    whitelist_values,
)
from .dispatch import DebouncedDispatcher
from .messages import split_message
from .runtime import AdapterRuntime, PROJECT_ROOT, SendText, SetTyping

__all__ = [
    "AdapterRuntime",
    "DebouncedDispatcher",
    "PROJECT_ROOT",
    "SendText",
    "SetTyping",
    "require_access_policy",
    "split_message",
    "whitelist_enabled",
    "whitelist_values",
]
