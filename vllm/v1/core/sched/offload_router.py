# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TokenCake Offload Router.

Routes KV cache offload to the appropriate storage tier
based on predicted function call duration and available backends.
"""

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class OffloadTarget(enum.IntEnum):
    HOST_RAM = 0       # Native OffloadingManager path
    UCM_LOCAL = 1      # UCM Store (NFS)
    UCM_REMOTE = 2     # UCM Store (Mooncake)


@dataclass
class OffloadRouterConfig:
    host_ram_threshold: float = 5.0      # < 5s → Host RAM
    ucm_remote_threshold: float = 60.0   # > 60s → UCM Remote


class OffloadRouter:
    """Routes offload to the appropriate storage tier.

    Selection strategy based on predict_time:
      predict_time < 5s   → Host RAM (fast in/out)
      predict_time 5-60s  → UCM Local Store (NFS, persistent)
      predict_time > 60s  → UCM Remote Store (Mooncake, cross-node)
    """

    def __init__(self, config: OffloadRouterConfig | None = None):
        self.config = config or OffloadRouterConfig()
        self.has_ucm = False
        self.has_host_offload = False

    def initialize(
        self,
        has_ucm_connector: bool = False,
        has_host_offload: bool = False,
    ) -> None:
        """Configure available offload paths based on deployment."""
        self.has_ucm = has_ucm_connector
        self.has_host_offload = has_host_offload

    def select_target(
        self,
        request: "Request",
        predict_time: float,
    ) -> OffloadTarget | None:
        """Select offload target based on predicted duration.

        Priority:
        1. predict_time < host_threshold → Host RAM (fastest)
        2. predict_time < ucm_remote_threshold → UCM Local
        3. predict_time >= ucm_remote_threshold → UCM Remote
        4. No available path → None (don't offload)
        """
        if predict_time < self.config.host_ram_threshold:
            if self.has_host_offload:
                return OffloadTarget.HOST_RAM
            elif self.has_ucm:
                return OffloadTarget.UCM_LOCAL
        elif predict_time < self.config.ucm_remote_threshold:
            if self.has_ucm:
                return OffloadTarget.UCM_LOCAL
            elif self.has_host_offload:
                return OffloadTarget.HOST_RAM
        else:
            if self.has_ucm:
                return OffloadTarget.UCM_REMOTE
            elif self.has_host_offload:
                return OffloadTarget.HOST_RAM

        return None
