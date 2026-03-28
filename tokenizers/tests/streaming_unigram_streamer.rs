use tokenizers::models::unigram::Unigram;
use tokenizers::streaming::{StreamTokenizer, PreTokenizerState, StreamingTokenizer};
use tokenizers::tokenizer::Model;

// Helper to create a simple Unigram model for testing
fn create_simple_unigram() -> Unigram {
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("hello".to_string(), 2.5),
        ("world".to_string(), 2.0),
        ("token".to_string(), 1.5),
        ("test".to_string(), 1.0),
    ];

    Unigram::from(vocab, Some(0), false).expect("Failed to build Unigram model")
}

fn baseline_unigram_tokens(
    model: &Unigram,
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
fn unigram_streamer_basic_creation() {
    let unigram = create_simple_unigram();
    let _streamer = tokenizers::streaming::UnigramStreamer::new(unigram);
    assert_eq!(
        <tokenizers::streaming::UnigramStreamer as StreamingTokenizer>::streaming_config().requires_word_boundaries,
        true
    );
    assert_eq!(
        <tokenizers::streaming::UnigramStreamer as StreamingTokenizer>::streaming_config().can_emit_incrementally,
        true
    );
}

#[test]
fn unigram_streamer_streams_with_word_boundaries() {
    let unigram = create_simple_unigram();
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::UnigramStreamer::new(unigram),
        1024,
    );

    orchestrator.process_chunk(b"hello world").unwrap();
    orchestrator.finalize().unwrap();

    let tokens = orchestrator.drain_tokens();
    assert!(!tokens.is_empty(), "Should produce tokens when word boundaries present");
}

#[test]
fn unigram_streamer_matches_baseline_unigram_tokenization() {
    let model_for_streaming = create_simple_unigram();
    let model_for_baseline = create_simple_unigram();
    let text = "hello world token test";

    // Baseline: non-streaming tokenization
    let baseline_tokens = baseline_unigram_tokens(&model_for_baseline, text);

    // Streaming: process and finalize
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::UnigramStreamer::new(model_for_streaming),
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
fn unigram_streamer_handles_chunks_at_word_boundary() {
    let model_for_streaming = create_simple_unigram();
    let model_for_baseline = create_simple_unigram();
    let text = "hello world";

    // Baseline
    let baseline_tokens = baseline_unigram_tokens(&model_for_baseline, text);

    // Streaming with chunk split at word boundary
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::UnigramStreamer::new(model_for_streaming),
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
