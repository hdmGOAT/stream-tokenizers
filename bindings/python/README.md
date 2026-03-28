# Tokenizers — Python Binding

**Fork of Hugging Face `tokenizers` with streaming support for Python.**

> This repository is an independently maintained fork of Hugging Face `tokenizers`,
> focused on streaming tokenization support. The Python binding includes streaming APIs
> under active development.

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

Bindings over the [Rust](https://github.com/huggingface/tokenizers/tree/master/tokenizers) implementation.
If you are interested in the High-level design, you can go check it there.

Otherwise, let's dive in!

## Main features:

 - Train new vocabularies and tokenize using 4 pre-made tokenizers (Bert WordPiece and the 3
   most common BPE versions).
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

### Installation (from source)

To build from source, you need Rust installed:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

Then build and install the Python binding:

```bash
cd bindings/python
python -m venv .env
source .env/bin/activate
pip install -e .
```

### Streaming Example

```python
from tokenizers import StreamingTokenizer

# Create a streaming tokenizer
tokenizer = StreamingTokenizer(model="path/to/model.json", buffer_size=4096)

# Feed chunks incrementally
tokenizer.process_chunk(b"chunk1")
tokenizer.process_chunk(b"chunk2")

# Finalize and drain tokens
tokenizer.finalize()
tokens = tokenizer.drain_tokens()
print(tokens)
```

### Load a pretrained tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-cased")
```

Whenever these provided tokenizers don't give you enough freedom, you can build your own tokenizer,
by putting all the different parts you need together.
You can check how we implemented the [provided tokenizers](https://github.com/huggingface/tokenizers/tree/master/bindings/python/py_src/tokenizers/implementations) and adapt them easily to your own needs.

#### Building a byte-level BPE

Here is an example showing how to build your own byte-level BPE by putting all the different pieces
together, and then saving it to a single file:

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(
    vocab_size=20000,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train([
    "./path/to/dataset/1.txt",
    "./path/to/dataset/2.txt",
    "./path/to/dataset/3.txt"
], trainer=trainer)

# And Save it
tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)
```

Now, when you want to use this tokenizer, this is as simple as:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("byte-level-bpe.tokenizer.json")

encoded = tokenizer.encode("I can feel the magic, can you?")
```

### Typing support and stub generation

The compiled PyO3 extension does not expose type annotations, so editors and type checkers would otherwise see most objects as `Any`. To provide full typing support, we use a two-step stub generation process:

1. **Rust introspection** (`tools/stub-gen/`): Uses `pyo3-introspection` to analyze the compiled extension and generate `.pyi` stub files
2. **Python enrichment** (`stub.py`): Adds docstrings from the runtime module and generates forwarding `__init__.py` shims

#### Running stub generation

The easiest way to regenerate stubs is via `make style`:

```bash
cd bindings/python
make style
```

This will:
1. Build the extension with `maturin develop --release`
2. Run introspection to generate `.pyi` files
3. Enrich stubs with docstrings via `stub.py`
4. Format with `ruff`

#### Running manually

To run the stub generator directly:

```bash
cd bindings/python
cargo run --manifest-path tools/stub-gen/Cargo.toml
python stub.py
```

The stub generator automatically:
- Builds the extension using maturin
- Copies the built `.so` to the project root for introspection
- Detects and sets `PYTHONHOME` for embedded Python (handles uv/venv environments)
- Generates stubs to `py_src/tokenizers/`

#### Troubleshooting

If you encounter Python initialization errors, you can manually set `PYTHONHOME`:

```bash
export PYTHONHOME=$(python3 -c 'import sys; print(sys.base_prefix)')
cargo run --manifest-path tools/stub-gen/Cargo.toml
```
