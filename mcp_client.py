"""Async JSON-RPC 2.0 client for the Rust `omni-orch` MCP server via stdio.

This transport spawns the Rust backend as a subprocess and communicates via
stdin/stdout, following the Model Context Protocol (MCP) standard for local tools.
Includes the MCP initialization handshake before any tool calls.
"""

from __future__ import annotations

import asyncio
import json
import itertools
import logging
from typing import Any, Optional, List

logger = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"

# Standard JSON-RPC error codes
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


class MCPError(RuntimeError):
    """A JSON-RPC error returned by the MCP server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPTransportError(RuntimeError):
    """A transport-layer failure (Process death, pipe breakage, malformed JSON)."""


class MCPClient:
    """Async JSON-RPC 2.0 client using stdio transport with MCP handshake."""

    def __init__(self, command: List[str]) -> None:
        """
        Args:
            command: The command to start the Rust MCP server,
                     e.g., ["/path/to/omnipulse-mcp"]
        """
        self.command = command
        self._process: Optional[asyncio.subprocess.Process] = None
        self._id_counter = itertools.count(1)
        self._lock = asyncio.Lock()  # Ensure one-at-a-time request/response over pipes

    async def start(self) -> None:
        """Spawn the Rust subprocess and complete MCP initialization handshake."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info(f"Spawned MCP backend: {' '.join(self.command)}")
        except Exception as e:
            raise ConnectionRefusedError(
                f"Failed to spawn Rust MCP backend at {self.command}: {e}"
            ) from e

        # MCP requires initialize → initialized handshake before any tool calls
        await self._initialize()

    async def _initialize(self) -> None:
        """Send initialize request, receive response, send notifications/initialized."""
        init_envelope = {
            "jsonrpc": JSONRPC_VERSION,
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "omni-agent",
                    "version": "1.0.0",
                },
            },
        }

        # Send initialize request
        request_json = json.dumps(init_envelope) + "\n"
        self._process.stdin.write(request_json.encode())
        await self._process.stdin.drain()

        # Read initialize response
        response_line = await self._process.stdout.readline()
        if not response_line:
            stderr_output = (await self._process.stderr.read()).decode()
            raise MCPTransportError(
                f"Rust backend closed stdout during initialization. Stderr: {stderr_output}"
            )

        body = json.loads(response_line)
        server_info = body.get("result", {}).get("serverInfo", {})
        logger.info(
            f"MCP initialized: name={server_info.get('name')} "
            f"version={server_info.get('version')}"
        )

        # Send notifications/initialized (fire-and-forget, no response expected)
        notif = {
            "jsonrpc": JSONRPC_VERSION,
            "method": "notifications/initialized",
            "params": {},
        }
        self._process.stdin.write((json.dumps(notif) + "\n").encode())
        await self._process.stdin.drain()

    async def stop(self) -> None:
        """Terminate the subprocess gracefully."""
        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

    async def __aenter__(self) -> "MCPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        """Send a tools/call JSON-RPC request and return the result."""
        if not self._process or self._process.returncode is not None:
            raise MCPTransportError("Rust MCP process is not running or has exited.")

        request_id = next(self._id_counter)
        envelope = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params,
            },
        }

        async with self._lock:
            # Write request
            request_json = json.dumps(envelope) + "\n"
            try:
                self._process.stdin.write(request_json.encode())
                await self._process.stdin.drain()
            except BrokenPipeError as e:
                raise MCPTransportError("Broken pipe: Rust backend crashed.") from e

            # Read response
            response_line = await self._process.stdout.readline()
            if not response_line:
                stderr_output = (await self._process.stderr.read()).decode()
                raise MCPTransportError(
                    f"Rust backend closed stdout unexpectedly. Stderr: {stderr_output}"
                )

        # Parse and validate
        try:
            body = json.loads(response_line)
        except ValueError as e:
            raise MCPTransportError(
                f"Malformed JSON from Rust: {response_line!r}"
            ) from e

        if body.get("jsonrpc") != JSONRPC_VERSION:
            raise MCPTransportError(f"Unexpected jsonrpc version: {body!r}")

        if body.get("id") != request_id:
            raise MCPTransportError(
                f"ID mismatch: expected {request_id}, got {body.get('id')!r}"
            )

        if "error" in body:
            err = body["error"]
            raise MCPError(
                code=err.get("code", JSONRPC_INTERNAL_ERROR),
                message=err.get("message", "unknown MCP error"),
                data=err.get("data"),
            )

        # MCP tool responses wrap the result in content[0].text as JSON
        result = body.get("result", {})
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except (ValueError, KeyError):
                return result
        return result

    async def generate_fingerprint(
        self,
        media_shm_name: str,
        sample_rate: int,
        config: dict,
    ) -> dict:
        """Zero-copy: passes only the shm segment name to the Rust backend."""
        return await self.call_tool(
            "generate_fingerprint",
            {
                "media_shm_name": media_shm_name,
                "signal_len": config.pop("signal_len", 344),
                "sample_rate": sample_rate,
                "config": config,
            },
        )

    async def compare_fingerprints(
        self,
        query_hash: str,
        reference_hash: Optional[str] = None,
    ) -> dict:
        """Run HNSW k-NN + Sliced Wasserstein against the registered corpus."""
        return await self.call_tool(
            "compare_fingerprints",
            {
                "query_hash": query_hash,
                "reference_hash": reference_hash,
            },
        )

    async def issue_license_token(
        self,
        fingerprint_hash: str,
        license_type: str,
    ) -> dict:
        """Mint an Ed25519-signed license token."""
        return await self.call_tool(
            "issue_license_token",
            {
                "fingerprint_hash": fingerprint_hash,
                "license_type": license_type,
            },
        )