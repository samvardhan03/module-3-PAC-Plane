"""POSIX shared-memory manager for the OmniPulse Phase 3 control plane.

PyArrow Plasma was removed upstream in pyarrow 12.0; this module replaces
it with the stdlib `multiprocessing.shared_memory.SharedMemory` primitive,
which on Linux is backed by `/dev/shm` tmpfs and on macOS/BSD by POSIX
`shm_open`. The C++/CUDA backend attaches to the same segment by name
via `shm_open(name, O_RDONLY)` and reads the float32 buffer directly —
no copy, no language-boundary serialisation.

The segment name is the SHA3-256[:20] hex digest of the raw buffer, so
ingest is content-addressed and idempotent across the cluster.
"""

from __future__ import annotations

import hashlib
import logging
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# 14 bytes (28 hex chars) of SHA3-256 digest is used as the segment name.
# macOS PSHMNAMLEN = 31 (incl. leading "/" and null terminator), capping
# usable hex at 29 chars; 28 fits everywhere and yields 112 bits of
# collision resistance — sufficient for a content-addressed cache.
SHM_OBJECT_ID_BYTES = 14


class SharedMemoryIngestionError(RuntimeError):
    """Raised when a tensor cannot be staged into POSIX shared memory."""


class SharedMemoryManager:
    """Content-addressed POSIX shared-memory ingest.

    Each ingested tensor is written into a live `SharedMemory` segment
    whose name is `sha3_256(buf).digest()[:20].hex()`. The segment is
    LEFT OPEN (not unlinked) on return so the C++/CUDA backend can attach
    to it by name. The owning process must call `release(name)` or
    `close_all()` to unlink — failure to do so leaks a tmpfs-backed page.
    """

    def __init__(self) -> None:
        self._open_blocks: Dict[str, shared_memory.SharedMemory] = {}

    def ingest_media_tensor(self, audio_array: np.ndarray) -> str:
        if not isinstance(audio_array, np.ndarray):
            raise TypeError(
                f"audio_array must be np.ndarray, got {type(audio_array).__name__}"
            )
        if audio_array.dtype != np.float32:
            raise TypeError(
                f"audio_array must be float32 for the WST ABI, got {audio_array.dtype}"
            )

        contiguous = np.ascontiguousarray(audio_array)
        raw = contiguous.tobytes()
        nbytes = len(raw)
        if nbytes == 0:
            raise SharedMemoryIngestionError("refusing to ingest a zero-length tensor")

        shm_name = hashlib.sha3_256(raw).digest()[:SHM_OBJECT_ID_BYTES].hex()

        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
        except FileExistsError:
            # Already in shared memory; re-attach for read but do not rewrite.
            existing = shared_memory.SharedMemory(name=shm_name, create=False)
            self._open_blocks.setdefault(shm_name, existing)
            logger.debug("shm block %s already present, reusing", shm_name)
            return shm_name

        # Direct memcpy into the live shm region — no intermediate buffer.
        np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf)[:] = np.frombuffer(
            raw, dtype=np.uint8
        )
        self._open_blocks[shm_name] = shm
        return shm_name

    def attach(self, shm_name: str) -> shared_memory.SharedMemory:
        """Open an existing segment by name without writing to it."""
        if shm_name in self._open_blocks:
            return self._open_blocks[shm_name]
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        self._open_blocks[shm_name] = shm
        return shm

    def get_buffer(self, shm_name: str) -> memoryview:
        return self.attach(shm_name).buf

    def release(self, shm_name: str) -> None:
        shm = self._open_blocks.pop(shm_name, None)
        if shm is None:
            return
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def close_all(self) -> None:
        for name in list(self._open_blocks.keys()):
            self.release(name)

    def __enter__(self) -> "SharedMemoryManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_all()
