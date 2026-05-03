"""Live integration test for the OmniPulse Python control plane.

ZERO MOCKS. This suite:

  1. Writes a real float32 tensor into a live POSIX shared-memory segment
     and re-attaches to it from a fresh handle to prove kernel residency.
  2. Issues a real HTTP request against the Rust MCP server at
     localhost:8080, exercising Yash's `omni-orch` backend end-to-end.
  3. Issues a real Anthropic API call, exercising the cognitive router
     against the live ANTHROPIC_API_KEY credential.

If the Rust backend is not alive on port 8080 the test fails immediately
with the message:
    "Live Rust backend not detected on port 8080. Failing integration test."
"""

from __future__ import annotations

import os
import socket
import sys
import wave
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

# Make the flat omni-agent source layout importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from control_plane import ControlPlane
from mcp_client import MCPClient
from shared_memory_manager import SharedMemoryManager

RUST_HOST = "localhost"
RUST_PORT = 8080
RUST_ENDPOINT = f"http://{RUST_HOST}:{RUST_PORT}"


def _rust_backend_alive() -> bool:
    try:
        with socket.create_connection((RUST_HOST, RUST_PORT), timeout=1.0):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _require_rust_backend() -> None:
    if not _rust_backend_alive():
        pytest.fail(
            "Live Rust backend not detected on port 8080. "
            "Failing integration test.",
            pytrace=False,
        )


def _require_anthropic_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail(
            "ANTHROPIC_API_KEY is not set in the live environment. "
            "Failing integration test.",
            pytrace=False,
        )


@pytest.fixture
def shm_manager():
    mgr = SharedMemoryManager()
    yield mgr
    mgr.close_all()


def test_shared_memory_real_kernel_write(shm_manager):
    """Real POSIX shm segment is created and visible to a fresh attacher."""
    rng = np.random.default_rng(seed=1337)
    tensor = rng.standard_normal(8192).astype(np.float32)

    shm_name = shm_manager.ingest_media_tensor(tensor)
    # SHA3-256 truncated to fit POSIX PSHMNAMLEN; must be valid hex.
    assert len(shm_name) >= 28
    bytes.fromhex(shm_name)

    # Independently re-attach via a brand-new handle. This proves the
    # segment lives in the kernel, not in process-local state.
    attached = shared_memory.SharedMemory(name=shm_name, create=False)
    try:
        view = np.ndarray(tensor.shape, dtype=np.float32, buffer=attached.buf)
        np.testing.assert_array_equal(view, tensor)
    finally:
        attached.close()


def test_shared_memory_is_content_addressed(shm_manager):
    """Identical tensors collapse to the same shm name (idempotent ingest)."""
    rng = np.random.default_rng(seed=99)
    tensor = rng.standard_normal(2048).astype(np.float32)

    first = shm_manager.ingest_media_tensor(tensor)
    second = shm_manager.ingest_media_tensor(tensor)
    assert first == second


@pytest.mark.asyncio
async def test_mcp_client_raises_connection_refused_on_dead_port():
    """A live client pointed at a closed port must raise ConnectionRefusedError."""
    # Port 1 is reserved and not bound to anything in normal environments.
    client = MCPClient(endpoint="http://127.0.0.1:1")
    try:
        with pytest.raises(ConnectionRefusedError):
            await client.call_tool("ping", {})
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_live_end_to_end_pipeline(tmp_path):
    """Full ingest → Rust MCP → Anthropic round trip against live services."""
    _require_rust_backend()
    _require_anthropic_key()

    # Synthesise a real PCM16 WAV so the default audio loader path runs.
    rng = np.random.default_rng(seed=2025)
    audio_f32 = (rng.standard_normal(44100) * 0.1).astype(np.float32)
    pcm16 = (audio_f32 * 32767.0).clip(-32768, 32767).astype(np.int16)
    wav_path = tmp_path / "live_input.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(pcm16.tobytes())

    plane = ControlPlane(
        shared_memory_manager=SharedMemoryManager(),
        mcp_client=MCPClient(endpoint=RUST_ENDPOINT),
    )

    try:
        # Live Anthropic call — proves the cognitive router is wired up.
        routed = plane.route_request(
            "Process this track and issue a derivative-commercial license."
        )
        assert isinstance(routed.get("raw"), str)
        assert routed["raw"].strip(), "Anthropic returned empty content"

        # Live full pipeline against the Rust backend.
        result = await plane.run_fingerprint_pipeline(
            str(wav_path),
            {
                "fingerprint_config": {
                    "J": 8, "Q": 16, "jtfs": True, "backend": "cuda",
                },
                "n_projections": 256,
                "license": {
                    "licensee_pubkey": "00" * 32,
                    "license_type": "derivative_commercial",
                    "expiry_unix": 9_999_999_999,
                    "royalty_basis_points": 500,
                },
            },
        )

        # Schema invariants from TDD §3.2.
        assert len(result["shm_name"]) >= 28
        assert "fingerprint_hash" in result["fingerprint"]
        assert "is_licensed_derivative" in result["comparison"]
        assert result["status"] in ("licensed", "not_derivative")
        if result["status"] == "licensed":
            assert "token_cid" in result["license"]
            assert result["license"]["token_cid"].startswith("bafk")
    finally:
        await plane.aclose()
