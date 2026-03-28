# Streamable Tokenizers

**Incremental, streaming tokenization for large-scale and low-latency LLM pipelines.**

This repository is an **independently maintained fork of Hugging Face `tokenizers`**, focused on **streaming support** and practical **Python / Node.js bindings** for modern inference workflows.

---

## Key Features

* **Streaming encode flows** — feed text incrementally via `process_chunk`, `finalize`, and `drain_tokens`
* **Low-memory processing** — handle large files or live streams without loading full input
* **Cross-platform bindings** — Python and Node.js interfaces for easy integration
* **Active Rust core** — efficient, safe, and production-ready tokenizer implementation

---

## Current State

* **Rust core library** — fully implemented and actively maintained
* **Python streaming binding** — implemented and tested
* **Node.js streaming binding** — implemented and tested
* Fully compatible with common subword models (BPE / Unigram)

---

## Repository Layout

```
tokenizers/          # Rust core library
bindings/python/     # Python bindings
bindings/node/       # Node.js bindings
docs/                # Documentation for streaming features
```

---

## Quick Start

### Python (from source)

```bash
cd bindings/python
python -m venv .env
source .env/bin/activate
pip install -e .
```

### Node.js (from source)

```bash
cd bindings/node
npm install
npm run build:debug
npm test -- lib/bindings
```

---

## Streaming Usage (Concept)

Create a **streaming tokenizer** from a model, feed chunks incrementally, finalize, and drain tokens as they are emitted.

**Python:**

```python
from streamable_tokenizers import StreamingTokenizer

tokenizer = StreamingTokenizer(model="bpe-en", buffer_size=4096)
for chunk in chunks:
    tokenizer.process_chunk(chunk)
tokenizer.finalize()
tokens = tokenizer.drain_tokens()
```

**Node.js:**

```js
const { StreamingTokenizer } = require('streamable-tokenizers');

const tokenizer = new StreamingTokenizer(model, { bufferSize: 4096 });
tokenizer.processChunk(chunk);
tokenizer.finalize();
const tokens = tokenizer.drainTokens();
```

---

## Why This Fork

* Optimized for **streaming tokenization** in real-time or large-scale pipelines
* Maintained **independently** of upstream roadmap
* Enables **incremental decoding** for constrained logit heads and low-memory inference

---

## Notes

* This project intentionally avoids upstream branding beyond attribution.
* Designed for **modular, high-performance LLM tooling**.