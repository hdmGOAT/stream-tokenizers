use ahash::AHashMap;
use tokenizers::models::wordlevel::WordLevelBuilder;
use tokenizers::tokenizer::Model;
use tokenizers::{streaming::{PreTokenizerState, StreamTokenizer, WordLevelStreamer}, Token};

fn build_wordlevel() -> tokenizers::models::wordlevel::WordLevel {
    let vocab: AHashMap<String, u32> = vec![
        ("<unk>".to_string(), 0_u32),
        ("hello".to_string(), 1_u32),
        (",".to_string(), 2_u32),
        ("world".to_string(), 3_u32),
        ("!".to_string(), 4_u32),
        ("token".to_string(), 5_u32),
        (":".to_string(), 6_u32),
    ]
    .into_iter()
    .collect();

    WordLevelBuilder::new()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .unwrap()
}

fn baseline_wordlevel_tokens(
    model: &tokenizers::models::wordlevel::WordLevel,
    input: &str,
) -> Vec<Token> {
    let mut state = PreTokenizerState::new();
    let pieces = state.process_chunk(input, true).unwrap();

    let mut tokens = Vec::new();
    for piece in pieces {
        tokens.extend(model.tokenize(&piece).unwrap());
    }
    tokens
}

#[test]
fn word_level_streamer_streams_in_chunks_and_emits_tokens() {
    let model = build_wordlevel();
    let mut streamer = StreamTokenizer::new(WordLevelStreamer::new(model), 1024);

    streamer.process_chunk(b"hello, wo").unwrap();
    let first_batch = streamer.drain_tokens();
    assert_eq!(first_batch.iter().map(|t| t.id).collect::<Vec<_>>(), vec![1, 2]);

    streamer.process_chunk(b"rld!").unwrap();
    let second_batch = streamer.drain_tokens();
    assert!(second_batch.is_empty());

    streamer.finalize().unwrap();
    let final_batch = streamer.drain_tokens();
    assert_eq!(final_batch.iter().map(|t| t.id).collect::<Vec<_>>(), vec![3, 4]);
}

#[test]
fn word_level_streamer_matches_baseline_wordlevel_tokenization() {
    let model_for_streaming = build_wordlevel();
    let model_for_baseline = build_wordlevel();

    let mut streamer = StreamTokenizer::new(WordLevelStreamer::new(model_for_streaming), 1024);
    streamer.process_chunk(b"hello, ").unwrap();
    streamer.process_chunk(b"world! token:").unwrap();
    streamer.finalize().unwrap();

    let streamed = streamer.drain_tokens();
    let baseline = baseline_wordlevel_tokens(&model_for_baseline, "hello, world! token:");

    let streamed_ids = streamed.iter().map(|token| token.id).collect::<Vec<_>>();
    let baseline_ids = baseline.iter().map(|token| token.id).collect::<Vec<_>>();
    assert_eq!(streamed_ids, baseline_ids);

    let streamed_values = streamed
        .iter()
        .map(|token| token.value.clone())
        .collect::<Vec<_>>();
    let baseline_values = baseline
        .iter()
        .map(|token| token.value.clone())
        .collect::<Vec<_>>();
    assert_eq!(streamed_values, baseline_values);
}

#[test]
fn word_level_streamer_unknown_tokens_match_baseline_unk_behavior() {
    let model_for_streaming = build_wordlevel();
    let model_for_baseline = build_wordlevel();

    let mut streamer = StreamTokenizer::new(WordLevelStreamer::new(model_for_streaming), 1024);
    streamer.process_chunk(b"mystery").unwrap();
    streamer.finalize().unwrap();
    let streamed = streamer.drain_tokens();

    let baseline = baseline_wordlevel_tokens(&model_for_baseline, "mystery");

    assert_eq!(streamed.len(), 1);
    assert_eq!(baseline.len(), 1);
    assert_eq!(streamed[0].id, baseline[0].id);
    assert_eq!(streamed[0].value, baseline[0].value);
}
