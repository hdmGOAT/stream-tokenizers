use tokenizers::models::bpe::{BPE, BpeBuilder};
use tokenizers::streaming::{StreamTokenizer, PreTokenizerState, StreamingTokenizer};
use tokenizers::tokenizer::Model;
use ahash::AHashMap;

// Helper to create a simple BPE model for testing
fn create_simple_bpe() -> BPE {
    let mut vocab = AHashMap::new();
    let tokens = vec!["<unk>", "l", "o", "w", "e", "r", "s", "d"];
    for (i, token) in tokens.iter().enumerate() {
        vocab.insert(token.to_string(), i as u32);
    }

    // Add merge pairs in order (priority)
    let merges = vec![
        ("l".to_string(), "o".to_string()),     // (0, 2) -> 8
        ("lo".to_string(), "w".to_string()),    // (8, 3) -> 9
        ("e".to_string(), "r".to_string()),     // (4, 5) -> 10
        ("er".to_string(), "s".to_string()),    // (10, 6) -> 11
    ];

    // Add merged tokens to vocab
    let merged_tokens = vec!["lo", "low", "er", "ers"];
    for (i, token) in merged_tokens.iter().enumerate() {
        vocab.insert(token.to_string(), (tokens.len() + i) as u32);
    }

    BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("Failed to build BPE model")
}

fn baseline_bpe_tokens(
    model: &BPE,
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
fn bpe_streamer_basic_creation() {
    let bpe = create_simple_bpe();
    let _streamer = tokenizers::streaming::BPEStreamer::new(bpe);
    assert_eq!(
        <tokenizers::streaming::BPEStreamer as StreamingTokenizer>::streaming_config().requires_word_boundaries,
        true
    );
    assert_eq!(
        <tokenizers::streaming::BPEStreamer as StreamingTokenizer>::streaming_config().can_emit_incrementally,
        true
    );
}

#[test]
fn bpe_streamer_streams_with_word_boundaries() {
    let bpe = create_simple_bpe();
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::BPEStreamer::new(bpe),
        1024,
    );

    // Use text with word boundaries
    orchestrator.process_chunk(b"lower world").unwrap();
    orchestrator.finalize().unwrap();

    let tokens = orchestrator.drain_tokens();
    assert!(!tokens.is_empty(), "Should produce tokens when word boundaries present");
}

#[test]
fn bpe_streamer_matches_baseline_bpe_tokenization() {
    let model_for_streaming = create_simple_bpe();
    let model_for_baseline = create_simple_bpe();
    let text = "lower world";

    // Baseline: non-streaming tokenization
    let baseline_tokens = baseline_bpe_tokens(&model_for_baseline, text);

    // Streaming: process and finalize
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::BPEStreamer::new(model_for_streaming),
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
fn bpe_streamer_handles_chunks_at_word_boundary() {
    let model_for_streaming = create_simple_bpe();
    let model_for_baseline = create_simple_bpe();
    let text = "lower world";

    // Baseline
    let baseline_tokens = baseline_bpe_tokens(&model_for_baseline, text);

    // Streaming with chunk split at word boundary
    let mut orchestrator = StreamTokenizer::new(
        tokenizers::streaming::BPEStreamer::new(model_for_streaming),
        1024,
    );

    orchestrator.process_chunk(b"lower ").unwrap();
    orchestrator.process_chunk(b"world").unwrap();
    orchestrator.finalize().unwrap();

    let streamed_tokens = orchestrator.drain_tokens();

    // Should match baseline
    let baseline_ids: Vec<_> = baseline_tokens.iter().map(|t| t.id).collect();
    let streamed_ids: Vec<_> = streamed_tokens.iter().map(|t| t.id).collect();
    
    assert_eq!(baseline_ids, streamed_ids);
}
