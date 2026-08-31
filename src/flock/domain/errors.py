"""Domain-level failures.

These are raised by value-object constructors. They signal that data violates a
promise the rest of the codebase relies on — never a recoverable condition.
"""


class InvariantError(ValueError):
    """A value object was handed data that breaks one of its invariants."""


class LeakageError(InvariantError):
    """Ground-truth data reached a computation that must not see it (spec R7)."""
