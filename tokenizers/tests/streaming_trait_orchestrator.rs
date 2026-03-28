use std::sync::{Arc, Mutex};

use tokenizers::{
    streaming::{StreamTokenizer, StreamingConfig, StreamingTokenizer},
    Split, Token,
};

#[derive(Clone)]
struct MockStreamer {
    state: Arc<Mutex<MockState>>,
}

#[derive(Default)]
struct MockState {
    process_calls: usize,
    finalize_calls: usize,
}

impl MockStreamer {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState::default())),
        }
    }
}

impl StreamingTokenizer for MockStreamer {
    fn streaming_config() -> StreamingConfig {
        StreamingConfig {
            requires_word_boundaries: true,
            lookahead_bytes: 32,
            can_emit_incrementally: true,
            min_chunk_size: 256,
        }
    }

    fn process_splits(
        &mut self,
        _splits: Vec<Split>,
        _is_final: bool,
    ) -> tokenizers::Result<Vec<Token>> {
        let mut state = self.state.lock().unwrap();
        state.process_calls += 1;
        Ok(vec![Token::new(10, "process".to_string(), (0, 0))])
    }

    fn finalize(&mut self) -> tokenizers::Result<Vec<Token>> {
        let mut state = self.state.lock().unwrap();
        state.finalize_calls += 1;
        Ok(vec![Token::new(20, "final".to_string(), (0, 0))])
    }
}

#[test]
fn stream_tokenizer_exposes_model_streaming_config() {
    let streamer = MockStreamer::new();
    let tokenizer = StreamTokenizer::new(streamer, 1024);

    let config = tokenizer.config();
    assert!(config.requires_word_boundaries);
    assert_eq!(config.lookahead_bytes, 32);
    assert!(config.can_emit_incrementally);
    assert_eq!(config.min_chunk_size, 256);
}

#[test]
fn stream_tokenizer_processes_chunks_and_emits_tokens() {
    let streamer = MockStreamer::new();
    let mut tokenizer = StreamTokenizer::new(streamer, 1024);

    tokenizer.process_chunk(b"hello, ").unwrap();
    let token = tokenizer.next_token().unwrap();
    assert_eq!(token.id, 10);
    assert_eq!(token.value, "process");
}

#[test]
fn stream_tokenizer_finalize_emits_remaining_tokens() {
    let streamer = MockStreamer::new();
    let mut tokenizer = StreamTokenizer::new(streamer, 1024);

    tokenizer.finalize().unwrap();
    let token = tokenizer.next_token().unwrap();
    assert_eq!(token.id, 20);
    assert_eq!(token.value, "final");
}

#[test]
fn stream_tokenizer_rejects_chunk_after_finalize() {
    let streamer = MockStreamer::new();
    let mut tokenizer = StreamTokenizer::new(streamer, 1024);

    tokenizer.finalize().unwrap();
    let error = tokenizer.process_chunk(b"late").unwrap_err();
    assert!(error.to_string().contains("finalize"));
}

#[derive(Clone, Default)]
struct SplitCountStreamer {
    received_split_counts: Arc<Mutex<Vec<usize>>>,
}

impl StreamingTokenizer for SplitCountStreamer {
    fn streaming_config() -> StreamingConfig {
        StreamingConfig {
            requires_word_boundaries: true,
            lookahead_bytes: 0,
            can_emit_incrementally: true,
            min_chunk_size: 1,
        }
    }

    fn process_splits(
        &mut self,
        splits: Vec<Split>,
        _is_final: bool,
    ) -> tokenizers::Result<Vec<Token>> {
        self.received_split_counts.lock().unwrap().push(splits.len());
        Ok(vec![Token::new(splits.len() as u32, "count".to_string(), (0, 0))])
    }

    fn finalize(&mut self) -> tokenizers::Result<Vec<Token>> {
        Ok(vec![])
    }
}

#[test]
fn stream_tokenizer_routes_chunk_through_pre_tokenizer_before_model() {
    let streamer = SplitCountStreamer::default();
    let mut tokenizer = StreamTokenizer::new(streamer, 1024);

    tokenizer.process_chunk(b"hello, world!").unwrap();
    let first = tokenizer.next_token().unwrap();
    assert_eq!(first.id, 2);

    tokenizer.finalize().unwrap();
    let second = tokenizer.next_token().unwrap();
    assert_eq!(second.id, 2);
}