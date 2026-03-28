use tokenizers::streaming::PreTokenizerState;

#[test]
fn pre_tokenizer_state_handles_word_and_punctuation_boundaries() {
    let mut state = PreTokenizerState::new();

    let tokens = state.process_chunk("hello, world!", false).unwrap();
    assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
}

#[test]
fn pre_tokenizer_state_handles_mid_token_chunk_splits() {
    let mut state = PreTokenizerState::new();

    assert_eq!(state.process_chunk("hel", false).unwrap(), Vec::<String>::new());
    assert_eq!(state.process_chunk("lo wo", false).unwrap(), vec!["hello"]);
    assert_eq!(state.process_chunk("rld", false).unwrap(), Vec::<String>::new());
    assert_eq!(state.finalize().unwrap(), vec!["world"]);
}

#[test]
fn pre_tokenizer_state_finalize_emits_trailing_pending_split() {
    let mut state = PreTokenizerState::new();

    assert_eq!(state.process_chunk("trailing", false).unwrap(), Vec::<String>::new());
    assert_eq!(state.finalize().unwrap(), vec!["trailing"]);
}