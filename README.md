<div align="center">
  <h1>omnipulse-agent</h1>
  <p><strong>The Agentic Control Plane for the OmniPulse Zero-Copy Hashing Architecture</strong></p>
  
  [![PyPI Version](https://img.shields.io/pypi/v/omnipulse-agent.svg)](https://pypi.org/project/omnipulse-agent/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
</div>

---

## ⚡️ Engineering Outcomes

`omnipulse-agent` is engineered for extreme performance. We are completely shifting the paradigm of how AI agents interact with high-performance backends.

*   **Zero-Copy Latency Reduction:** Eliminate JSON serialization overhead entirely. 
*   **Memory-Safe Cross-Language Architecture:** Seamlessly route LLM commands from Python to a Rust/C++ backend safely.
*   **15+ GB/s DMA Transfer Potential:** Utilize POSIX Shared Memory and Apache Arrow to hit memory bandwidth limits, leaving network bottlenecks in the dust.
*   **Direct-to-Metal Dispatch:** Route LLM intelligence straight into raw compute via our optimized `stdio` pipelines.

## 🚀 Installation

```bash
pip install omnipulse-agent
```

## 💻 Quickstart

Below is a minimal example demonstrating how to initialize the Anthropic client within the OmniPulse ecosystem and execute a zero-copy task, like generating a fingerprint.

```python
import os
import asyncio
from omnipulse_agent.mcp_client import MCPClient
from omnipulse_agent.control_plane import AgenticControlPlane
from anthropic import AsyncAnthropic

async def main():
    # 1. Initialize the Async Anthropic Client
    anthropic = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # 2. Setup the OmniPulse Agentic Control Plane
    control_plane = AgenticControlPlane(
        anthropic_client=anthropic,
        model="claude-3-opus-20240229",
        backend_path="/path/to/rust/backend"
    )

    # 3. Connect and execute
    await control_plane.start()
    
    # 4. Route a command directly (Zero-Copy)
    result = await control_plane.execute_command(
        command="generate_fingerprint",
        data_payload={"region": "us-east", "intensity": "high"}
    )
    
    print(f"Zero-copy execution completed! Result: {result}")
    
    await control_plane.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗 Architecture Highlight

Traditional LLM agentic frameworks serialize data to JSON and send it over HTTP/gRPC, choking the network stack and wasting CPU cycles.

**OmniPulse** does things differently:
1. **Apache Arrow Memory Format:** Data remains in an in-memory columnar format that both Python and Rust/C++ understand natively.
2. **POSIX Shared Memory:** Instead of sending payloads over sockets, we pass pointers to shared memory regions. 
3. **Rust `stdio` Pipe:** Control commands are dispatched over lightweight, ultra-fast `stdio` pipes directly to the compute engine.

The result? Multi-gigabyte LLM context windows and large embedding arrays mapped into the backend's memory space instantly.

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for more details.
