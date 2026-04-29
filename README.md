# wrapper-llama

A lightweight Python wrapper to manage the lifecycle of **llama-server** (from [llama.cpp](https://github.com/ggml-org/llama.cpp)). It handles process spawning, health checks, and clean shutdown — so you can focus on building your LLM application.

## Features

- Start and stop `llama-server` from Python
- Automatic health check polling until the server is ready
- Context manager (`with` statement) for guaranteed cleanup
- `atexit` safety net when not using the context manager
- Zero external dependencies for the core library
- Cross-platform (Windows and Linux)

## Installation

```bash
pip install wrapper-llama
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add wrapper-llama
```

## Quick Start

```python
from wrapper_llama import LlamaServerManager
import urllib.request
import json

with LlamaServerManager(
    exe_path="llama-server",
    model_path="/path/to/your/model.gguf",
    gpu_layers=999,  # full GPU offload
) as server:

    url = f"http://{server.host}:{server.port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {"role": "user", "content": "Tell me a short joke."}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
        print(data["choices"][0]["message"]["content"])
```

## API Reference

### `LlamaServerManager`

```python
LlamaServerManager(
    exe_path: str,
    model_path: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    gpu_layers: int = 0,
    ctx_size: int = 2048,
    threads: Optional[int] = None,
    timeout_start: float = 30.0,
    debug: bool = False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exe_path` | `str` | **required** | Path to `llama-server` executable |
| `model_path` | `str` | **required** | Path to the `.gguf` model file |
| `port` | `int` | `8080` | Port to bind the server to |
| `host` | `str` | `"127.0.0.1"` | Host address to bind to |
| `gpu_layers` | `int` | `0` | Number of GPU layers (`0` = CPU, `999` = full GPU) |
| `ctx_size` | `int` | `2048` | Context window size |
| `threads` | `Optional[int]` | `None` | Number of threads (defaults to CPU count) |
| `timeout_start` | `float` | `30.0` | Max seconds to wait for server readiness |
| `debug` | `bool` | `False` | Show server stdout/stderr when `True` |

#### Methods

- **`start()`** — Launch the server and wait until the `/health` endpoint responds.
- **`stop()`** — Gracefully terminate the server (with `kill` fallback after 5s).
- **`is_running() -> bool`** — Check if the server is alive and responsive.

#### Context Manager

```python
with LlamaServerManager(...) as server:
    # server is ready here
    pass
# server.stop() called automatically
```

## Running the Example

1. Copy `.env.example` to `.env` and fill in your paths:

```env
LLAMA_EXE_PATH=llama-server
LLAMA_MODEL_PATH=/path/to/your/model.gguf
LLAMA_GPU_LAYERS=0
```

2. Install dev dependencies and run:

```bash
uv sync --extra dev
uv run python examples/basic_chat.py
```

## Development

```bash
# Set up environment
uv sync --extra dev

# Build
uv build
```

## License

MIT
