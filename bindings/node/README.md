# Tokenizers — Node.js Binding

**Fork of Hugging Face `tokenizers` with streaming support for Node.js.**

> This repository is an independently maintained fork of Hugging Face `tokenizers`,
> focused on streaming tokenization support. The Node.js binding includes streaming APIs
> under active development.

## Main features

 - Train new vocabularies and tokenize using 4 pre-made tokenizers (Bert WordPiece and the 3
   most common BPE versions).
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

## Installation (from source)

```bash
cd bindings/node
npm install
npm run build:debug
```

## Streaming Example

```ts
import { StreamingTokenizer } from "tokenizers";

const tokenizer = new StreamingTokenizer(model);

// Feed chunks incrementally
tokenizer.processChunk(Buffer.from("chunk1"));
tokenizer.processChunk(Buffer.from("chunk2"));

// Finalize and drain tokens
tokenizer.finalize();
const tokens = tokenizer.drainTokens();
console.log(tokens);
```

## License

[Apache License 2.0](../../LICENSE)
