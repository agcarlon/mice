from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Recorder:
    """
    Minimal event recorder for MICE.

    Keep this extremely lightweight; convert to pandas on demand (future).
    """

    events: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        event: str,
        num_grads: int,
        hier_length: int,
        last_v: Optional[float],
        grad_norm: Optional[float],
        iteration: int,
        terminate_reason: Optional[str] = None,
    ) -> None:
        self.events.append(
            {
                "event": event,
                "num_grads": num_grads,
                "vl": last_v,
                "grad_norm": grad_norm,
                "hier_length": hier_length,
                "iteration": iteration,
                "terminate_reason": terminate_reason,
            }
        )

    def as_list(self):
        return self.events
