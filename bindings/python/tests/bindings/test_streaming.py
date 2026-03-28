from tokenizers import StreamingTokenizer as RootStreamingTokenizer
from tokenizers.models import WordLevel
from tokenizers.streaming import StreamingTokenizer as SubmoduleStreamingTokenizer


class TestStreamingTokenizer:
    def test_import_paths(self):
        assert RootStreamingTokenizer is not None
        assert SubmoduleStreamingTokenizer is not None

    def test_wordlevel_streaming_roundtrip(self):
        model = WordLevel(vocab={"Hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]")
        stream = SubmoduleStreamingTokenizer(model, buffer_size=1024)

        config = stream.config()
        assert isinstance(config, dict)
        assert {
            "requires_word_boundaries",
            "lookahead_bytes",
            "can_emit_incrementally",
            "min_chunk_size",
        }.issubset(config.keys())

        stream.process_chunk(b"Hello ")
        stream.process_chunk(b"world")
        stream.finalize()

        tokens = stream.drain_tokens()
        assert isinstance(tokens, list)
        assert len(tokens) >= 2
        assert all(isinstance(token, dict) for token in tokens)
        assert all({"id", "value", "offsets"}.issubset(token.keys()) for token in tokens)

        values = [token["value"] for token in tokens]
        assert values == ["Hello", "world"]

        assert stream.drain_tokens() == []
