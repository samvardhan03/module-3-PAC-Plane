"""
End-to-end test script for OmniPulse stack.
Bridges the control plane against the Rust MCP server.
"""
import asyncio
import logging
import numpy as np
from mcp_client import MCPClient
from shared_memory_manager import SharedMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BINARY = "/Users/yashmishra/Work/rwst/omnipulse-rs/target/release/omnipulse-mcp"

SIGNAL_LEN = 344

# Config that matches Rust server's WstConfig field names
RUST_CONFIG = {
    "j": 8,
    "q": 16,
    "depth": 2,
    "jtfs": False,
    "dim": 1,
}


async def main():
    shm = SharedMemoryManager()

    async with MCPClient(command=[BINARY]) as client:

        # --- Signal A: zeros ---
        logger.info("Creating signal A (zeros)...")
        signal_a = np.zeros(SIGNAL_LEN, dtype=np.float32)
        shm_name_a = shm.ingest_media_tensor(signal_a)
        logger.info(f"Signal A in shm: {shm_name_a}")

        result_a = await client.generate_fingerprint(
            media_shm_name=shm_name_a,
            sample_rate=44100,
            config={**RUST_CONFIG, "signal_len": SIGNAL_LEN},
        )
        logger.info(f"Fingerprint A: {result_a}")
        fp_id_a = result_a.get("fingerprint_id")
        if not fp_id_a:
            raise RuntimeError(f"No fingerprint_id in response: {result_a}")

        # --- Signal B: sine wave ---
        logger.info("Creating signal B (sine)...")
        signal_b = np.sin(np.arange(SIGNAL_LEN, dtype=np.float32) * 0.1)
        shm_name_b = shm.ingest_media_tensor(signal_b)
        logger.info(f"Signal B in shm: {shm_name_b}")

        result_b = await client.generate_fingerprint(
            media_shm_name=shm_name_b,
            sample_rate=44100,
            config={**RUST_CONFIG, "signal_len": SIGNAL_LEN},
        )
        logger.info(f"Fingerprint B: {result_b}")
        fp_id_b = result_b.get("fingerprint_id")
        if not fp_id_b:
            raise RuntimeError(f"No fingerprint_id in response: {result_b}")

        # --- Compare A against index ---
        logger.info(f"Comparing {fp_id_a} against index...")
        comparison = await client.compare_fingerprints(
            query_hash=fp_id_a,
        )
        logger.info(f"Comparison result: {comparison}")

        # --- Summary ---
        print("\n=== OmniPulse End-to-End Test ===")
        print(f"Fingerprint A (zeros): {fp_id_a}")
        print(f"Fingerprint B (sine):  {fp_id_b}")
        print(f"Neighbors of A:")
        for n in comparison.get("neighbors", []):
            print(f"  {n.get('fingerprint_id')}  distance={n.get('distance')}")

    shm.close_all()


if __name__ == "__main__":
    asyncio.run(main())