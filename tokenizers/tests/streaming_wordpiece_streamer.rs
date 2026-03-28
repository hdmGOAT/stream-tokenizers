use tokenizers::models::wordpiece::{WordPiece, WordPieceBuilder};
use tokenizers::streaming::{StreamTokenizer, PreTokenizerState, StreamingTokenizer};
use tokenizers::tokenizer::Model;
use ahash::AHashMap;

// Helper to create a simple WordPiece model for testing
fn create_simple_wordpiece() -> WordPiece {
    let mut vocab = AHashMap::new();
    let tokens = vec![
        "[UNK]", "hello", "world", ",", "!", "token", ":", "##ing", "##ed", "##er",
    ];
    for (i, token) in tokens.iter().enumerate() {
        vocab.insert(token.to_string(), i as u32);
    }

    WordPieceBuilder::new()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .continuing_subword_prefix("##".to_string())
        .build()
        .expect("Failed to build WordPiece model")
}

fn baseline_wordpiece_tokens(
    model: &WordPiece,
    input: &str,
) -> Vec<tokenizers::Token> {
    let mut state = PreTokenizerState::new();
    let pieces = state.process_chunk(input, true).unwrap();

    let mut tokens = Vec::new();
    for piece in pieces {
        tokens.extend(model.tokenize(&piece).unwrap());
    }
    tokens
}

#[test]
fn wordpiece_streamer_basic_creation() {
    let wp = create_simple_wordpiece();
    let _streamer = tokenizers::streaming::WordPieceStreamer::new(wp);
    assert_eq!(
        <tokenizers::streaming::WordPieceStreamer as StreamingTokenizer>::streaming_config().requires_word_boundaries,
        true
    );
    assert_eq!(
        <tokenizers::streaming::WordPieceStreamer as StreamingTokenizer>::streaming_config().can_emit_incrementally,
        true
    );
}

#[test]
fn wordpiece_streamer_streams_with_word_boundaries() {
    let wp = create_simple_wordpiece();
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::WordPieceStreamer::new(wp),
        1024,
    );

    orchestrator.process_chunk(b"hello world").unwrap();
    orchestrator.finalize().unwrap();

    let tokens = orchestrator.drain_tokens();
    assert!(!tokens.is_empty(), "Should produce tokens when word boundaries present");
}

#[test]
fn wordpiece_streamer_matches_baseline_wordpiece_tokenization() {
    let model_for_streaming = create_simple_wordpiece();
    let model_for_baseline = create_simple_wordpiece();
    let text = "hello world , token";

    // Baseline: non-streaming tokenization
    let baseline_tokens = baseline_wordpiece_tokens(&model_for_baseline, text);

    // Streaming: process and finalize
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::WordPieceStreamer::new(model_for_streaming),
        1024,
    );

    orchestrator.process_chunk(text.as_bytes()).unwrap();
    orchestrator.finalize().unwrap();

    let streamed_tokens = orchestrator.drain_tokens();

    // Compare token IDs
    let baseline_ids: Vec<_> = baseline_tokens.iter().map(|t| t.id).collect();
    let streamed_ids: Vec<_> = streamed_tokens.iter().map(|t| t.id).collect();

    assert_eq!(
        baseline_ids,
        streamed_ids,
        "Token IDs mismatch: baseline {:?} vs streamed {:?}",
        baseline_ids,
        streamed_ids
    );
}

#[test]
fn wordpiece_streamer_handles_chunks_at_word_boundary() {
    let model_for_streaming = create_simple_wordpiece();
    let model_for_baseline = create_simple_wordpiece();
    let text = "hello world";

    // Baseline
    let baseline_tokens = baseline_wordpiece_tokens(&model_for_baseline, text);

    // Streaming with chunk split at word boundary
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::WordPieceStreamer::new(model_for_streaming),
        1024,
    );

    orchestrator.process_chunk(b"hello ").unwrap();
    orchestrator.process_chunk(b"world").unwrap();
    orchestrator.finalize().unwrap();

    let streamed_tokens = orchestrator.drain_tokens();

    // Should match baseline
    let baseline_ids: Vec<_> = baseline_tokens.iter().map(|t| t.id).collect();
    let streamed_ids: Vec<_> = streamed_tokens.iter().map(|t| t.id).collect();

    assert_eq!(baseline_ids, streamed_ids);
}
