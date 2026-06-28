"""Base classes for the self-adjointness filter pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from openfermion import BosonOperator

FilterResult = Optional[bool]


class SelfAdjointFilter(ABC):
    """Abstract filter in the self-adjointness chain of responsibility."""

    def __init__(self) -> None:
        self._next: Optional[SelfAdjointFilter] = None

    def set_next(self, nxt: SelfAdjointFilter) -> SelfAdjointFilter:
        """Attach the next filter and return it for chaining."""
        self._next = nxt
        return nxt

    def handle(self, H: BosonOperator) -> FilterResult:
        """Run this filter; delegate to the next on inconclusive results."""
        result = self.evaluate(H)
        if result is not None:
            return result
        if self._next is not None:
            return self._next.handle(H)
        return None

    @abstractmethod
    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``True``, ``False``, or ``None`` (inconclusive)."""
