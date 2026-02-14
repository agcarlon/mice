from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class DropRestartClipPolicy:
    """
    Policy parameters.

    Drop is checked every iteration. Clip can be checked every `clip_every`
    iterations or fully deactivated (clip_type=None).
    """

    drop_param: float = 0.5
    restart_param: float = 0.0
    max_hierarchy_size: int = 1000
    aggr_cost: float = 0.1

    # None disables clipping by default.
    clip_type: Optional[str] = None
    clip_every: int = 0  # 0 means "never on schedule" (only triggered by other conditions)

