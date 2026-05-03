"""LLM agentic control plane for OmniPulse (Phase 3).

Claude (Anthropic) acts as a cognitive router: it parses operator
requests and selects the deterministic MCP tool sequence to execute, but
never touches tensor data and never fabricates shm names. All raw audio
crosses process boundaries exclusively via POSIX shared memory.

The Anthropic client is constructed against the live `ANTHROPIC_API_KEY`
environment variable; if that variable is unset the control plane refuses
to start rather than silently fall back to a broken state.
"""

from __future__ import annotations

import logging
import os
import time
import wave
from typing import Any, Optional

import anthropic
import numpy as np

from mcp_client import MCPClient
from shared_memory_manager import SharedMemoryManager

logger = logging.getLogger(__name__)

# Per OmniPulse policy, default to the latest Claude.
DEFAULT_MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are the OmniPulse cognitive router.

You map natural-language operator requests to a deterministic sequence of
MCP tool invocations. You never directly manipulate tensor data.

You are FORBIDDEN from fabricating these parameters; they MUST originate
from a prior tool response in this same workflow:

  * media_shm_name    — the hex POSIX shared-memory segment name returned
                        by SharedMemoryManager.ingest_media_tensor.
  * fingerprint_hash  — the SHA3-256 hex digest returned by
                        generate_fingerprint.
  * comparison_result_id / audit_id — returned by compare_fingerprints.

If any required identifier is missing, surface the gap rather than guess.
"""


def _load_audio_as_float32(path: str) -> np.ndarray:
    """Load a PCM16 WAV file into a normalised float32 mono array."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"only PCM16 wav supported, got sampwidth={sampwidth}")

    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def _build_live_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set; the live cognitive router cannot start"
        )
    return anthropic.Anthropic(api_key=api_key)


class ControlPlane:
    """Python agentic control plane.

    Composes the SharedMemoryManager (memory bridge), MCPClient (Rust
    transport), and an Anthropic client (cognitive router) into a single
    orchestration surface.
    """

    def __init__(
        self,
        *,
        shared_memory_manager: Optional[SharedMemoryManager] = None,
        mcp_client: Optional[MCPClient] = None,
        anthropic_client: Optional[anthropic.Anthropic] = None,
        model: str = DEFAULT_MODEL,
        audio_loader=_load_audio_as_float32,
    ) -> None:
        self.shm = shared_memory_manager or SharedMemoryManager()
        self.mcp = mcp_client or MCPClient()
        self.llm = anthropic_client or _build_live_anthropic_client()
        self.model = model
        self._load_audio = audio_loader

    async def aclose(self) -> None:
        await self.mcp.aclose()
        self.shm.close_all()

    def route_request(self, natural_language_request: str) -> dict:
        """Use the live Anthropic API to extract structured parameters."""
        response = self.llm.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": natural_language_request}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return {"raw": text}

    async def run_fingerprint_pipeline(
        self,
        audio_path: str,
        config: dict,
    ) -> dict:
        """Execute the full ingest → fingerprint → compare → license workflow."""

        # Step 1: ingest tensor into POSIX shared memory. The tensor itself
        # never crosses an MCP boundary — only the 40-char hex shm name does.
        audio = self._load_audio(audio_path)
        shm_name = self.shm.ingest_media_tensor(audio)
        logger.info("ingested %s -> shm_name=%s", audio_path, shm_name)

        # Step 2: fingerprint via the Rust + C++/CUDA backend.
        fp_config = config.get(
            "fingerprint_config",
            {"J": 8, "Q": 16, "jtfs": True, "backend": "cuda"},
        )
        fingerprint = await self.mcp.generate_fingerprint(
            media_shm_name=shm_name,
            config=fp_config,
        )
        if "fingerprint_hash" not in fingerprint:
            raise RuntimeError(
                f"generate_fingerprint missing fingerprint_hash: {fingerprint!r}"
            )

        # Step 3: HNSW k-NN + Sliced Wasserstein comparison.
        comparison = await self.mcp.compare_fingerprints(
            query_hash=fingerprint["fingerprint_hash"],
            reference_hash=config.get("reference_hash"),
            n_projections=config.get("n_projections", 256),
        )

        result: dict[str, Any] = {
            "shm_name": shm_name,
            "fingerprint": fingerprint,
            "comparison": comparison,
            "license": None,
            "status": "not_derivative",
        }

        # Step 4: conditional licensing — gated on the Rust comparator,
        # never on LLM reasoning.
        if comparison.get("is_licensed_derivative"):
            license_cfg = config.get("license", {})
            if "licensee_pubkey" not in license_cfg:
                raise ValueError(
                    "license.licensee_pubkey is required to issue a derivative token"
                )
            token = await self.mcp.issue_license_token(
                fingerprint_hash=fingerprint["fingerprint_hash"],
                comparison_result_id=comparison.get("audit_id", ""),
                licensee_pubkey=license_cfg["licensee_pubkey"],
                license_type=license_cfg.get("license_type", "derivative_commercial"),
                expiry_unix=license_cfg.get(
                    "expiry_unix", int(time.time()) + 365 * 24 * 3600
                ),
                royalty_basis_points=license_cfg.get("royalty_basis_points", 500),
            )
            result["license"] = token
            result["status"] = "licensed"

        return result
