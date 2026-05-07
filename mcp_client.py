"""Async JSON-RPC 2.0 client for the Rust `omni-orch` MCP server via stdio.

This transport spawns the Rust backend as a subprocess and communicates via 
stdin/stdout, following the Model Context Protocol (MCP) standard for local tools.
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
    """Async JSON-RPC 2.0 client using stdio transport."""

    def __init__(self, command: List[str]) -> None:
        """
        Args:
            command: The command to start the Rust MCP server, 
                     e.g., ["cargo", "run", "--release"]
        """
        self.command = command
        self._process: Optional[asyncio.subprocess.Process] = None
        self._id_counter = itertools.count(1)
        self._lock = asyncio.Lock()  # Ensure one-at-a-time request/response logic over pipes

    async def start(self) -> None:
        """Spawn the Rust subprocess."""
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

    async def stop(self) -> None:
        """Terminate the subprocess."""
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
        if not self._process or self._process.returncode is not None:
            raise MCPTransportError("Rust MCP process is not running or has exited.")

        request_id = next(self._id_counter)
        envelope = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": tool_name,
            "params": params,
        }

        async with self._lock:
            # 1. Write the request as a single line
            request_json = json.dumps(envelope) + "\n"
            try:
                self._process.stdin.write(request_json.encode())
                await self._process.stdin.drain()
            except BrokenPipeError as e:
                raise MCPTransportError("Broken pipe: Rust backend crashed.") from e

            # 2. Read the response line
            response_line = await self._process.stdout.readline()
            if not response_line:
                # Read stderr to see why it died
                stderr_output = (await self._process.stderr.read()).decode()
                raise MCPTransportError(f"Rust backend closed stdout unexpectedly. Stderr: {stderr_output}")

        # 3. Parse and validate
        try:
            body = json.loads(response_line)
        except ValueError as e:
            raise MCPTransportError(f"Malformed JSON from Rust: {response_line!r}") from e

        if body.get("jsonrpc") != JSONRPC_VERSION:
            raise MCPTransportError(f"Unexpected jsonrpc version: {body!r}")
        
        if body.get("id") != request_id:
            raise MCPTransportError(f"ID mismatch: expected {request_id}, got {body.get('id')!r}")

        if "error" in body:
            err = body["error"]
            raise MCPError(
                code=err.get("code", JSONRPC_INTERNAL_ERROR),
                message=err.get("message", "unknown MCP error"),
                data=err.get("data"),
            )

        return body.get("result")

    async def generate_fingerprint(self, media_shm_name: str, sample_rate: int, config: dict) -> dict:
        """Zero-copy: Passes only the shm segment name to the Rust backend."""
        return await self.call_tool(
            "generate_fingerprint",
            {
                "media_shm_name": media_shm_name, 
                "sample_rate": sample_rate, 
                "config": config
            },
        )

    async def compare_fingerprints(self, query_hash: str, reference_hash: Optional[str] = None) -> dict:
        return await self.call_tool(
            "compare_fingerprints",
            {"query_hash": query_hash, "reference_hash": reference_hash},
        )

    async def issue_license_token(self, fingerprint_hash: str, license_type: str) -> dict:
        return await self.call_tool(
            "issue_license_token",
            {"fingerprint_hash": fingerprint_hash, "license_type": license_type},
        )
