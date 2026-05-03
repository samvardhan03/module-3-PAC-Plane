"""Async JSON-RPC 2.0 client for the Rust `omni-orch` MCP server.

This is the wire-level transport between the Python control plane and the
Rust orchestrator. The three tool wrappers exposed here mirror the schemas
documented in OmniPulse TDD §3.2 and are the only sanctioned channel for
fingerprint generation, comparison, and licence issuance.

Connection failures to the Rust backend are surfaced as the canonical
`ConnectionRefusedError` so callers (and CI) can immediately distinguish
"backend down" from any other failure mode.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MCP_ENDPOINT = "http://localhost:8080"
JSONRPC_VERSION = "2.0"

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

OMNIPULSE_SHM_NOT_FOUND = -32001
OMNIPULSE_LICENSE_GATE_FAILED = -32002


class MCPError(RuntimeError):
    """A JSON-RPC error returned by the MCP server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPTransportError(RuntimeError):
    """A transport-layer failure (HTTP, TLS, malformed envelope)."""


class MCPClient:
    """Async JSON-RPC 2.0 client for the Rust `omni-orch` MCP server."""

    def __init__(
        self,
        endpoint: str = DEFAULT_MCP_ENDPOINT,
        *,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.endpoint = endpoint
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._id_counter = itertools.count(1)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "MCPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        request_id = next(self._id_counter)
        envelope = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": tool_name,
            "params": params,
        }

        try:
            response = await self._client.post(self.endpoint, json=envelope)
        except httpx.ConnectError as e:
            # Surface as the canonical OS-level exception so CI/operators
            # immediately recognise "the Rust backend is down".
            raise ConnectionRefusedError(
                f"Rust MCP backend at {self.endpoint} is not reachable: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise MCPTransportError(f"transport error to {self.endpoint}: {e}") from e

        if response.status_code != 200:
            raise MCPTransportError(
                f"non-200 from MCP server: {response.status_code} {response.text!r}"
            )

        try:
            body = response.json()
        except ValueError as e:
            raise MCPTransportError(f"non-JSON response: {response.text!r}") from e

        if body.get("jsonrpc") != JSONRPC_VERSION:
            raise MCPTransportError(f"unexpected jsonrpc field in response: {body!r}")
        if body.get("id") != request_id:
            raise MCPTransportError(
                f"id mismatch: expected {request_id}, got {body.get('id')!r}"
            )

        if "error" in body:
            err = body["error"]
            raise MCPError(
                code=err.get("code", JSONRPC_INTERNAL_ERROR),
                message=err.get("message", "unknown MCP error"),
                data=err.get("data"),
            )

        if "result" not in body:
            raise MCPTransportError(f"missing result field in response: {body!r}")

        return body["result"]

    async def generate_fingerprint(
        self,
        media_shm_name: str,
        config: dict,
    ) -> dict:
        """Invoke the Rust+CUDA WST/JTFS pipeline against a shm-staged tensor."""
        return await self.call_tool(
            "generate_fingerprint",
            {"media_shm_name": media_shm_name, "config": config},
        )

    async def compare_fingerprints(
        self,
        query_hash: str,
        reference_hash: Optional[str] = None,
        n_projections: int = 256,
    ) -> dict:
        """Run HNSW k-NN + Sliced Wasserstein against the registered corpus."""
        return await self.call_tool(
            "compare_fingerprints",
            {
                "query_hash": query_hash,
                "reference_hash": reference_hash,
                "n_projections": n_projections,
            },
        )

    async def issue_license_token(
        self,
        fingerprint_hash: str,
        comparison_result_id: str,
        licensee_pubkey: str,
        license_type: str,
        expiry_unix: int,
        royalty_basis_points: int,
    ) -> dict:
        """Mint an Ed25519-signed, IPFS-pinned licence token."""
        return await self.call_tool(
            "issue_license_token",
            {
                "fingerprint_hash": fingerprint_hash,
                "comparison_result_id": comparison_result_id,
                "licensee_pubkey": licensee_pubkey,
                "license_type": license_type,
                "expiry_unix": expiry_unix,
                "royalty_basis_points": royalty_basis_points,
            },
        )
