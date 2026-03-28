use tokenizers::{streaming::NormalizerState, NormalizedString};

fn nfc(input: &str) -> String {
    let mut normalized = NormalizedString::from(input);
    normalized.nfc();
    normalized.get().to_string()
}

#[test]
fn normalizer_state_handles_combining_marks_across_chunks() {
    let mut state = NormalizerState::new();

    let chunk_1 = state.process_chunk("e", false).unwrap();
    let chunk_2 = state.process_chunk("\u{0301}", false).unwrap();
    let tail = state.finalize().unwrap();

    assert_eq!(chunk_1, "");
    assert_eq!(chunk_2, "");
    assert_eq!(tail, "é");
}

#[test]
fn normalizer_state_finalize_flushes_pending_text() {
    let mut state = NormalizerState::new();

    let out = state.process_chunk("hello", false).unwrap();
    let tail = state.finalize().unwrap();

    assert_eq!(out, "hell");
    assert_eq!(tail, "o");
}

#[test]
fn normalizer_state_matches_non_streaming_nfc_equivalence() {
    let input = "Cafe\u{0301} de\u{0301}ja\u{0300} vu!";
    let expected = nfc(input);

    let mut state = NormalizerState::new();
    let mut output = String::new();

    output.push_str(&state.process_chunk("Ca", false).unwrap());
    output.push_str(&state.process_chunk("fe", false).unwrap());
    output.push_str(&state.process_chunk("\u{0301} d", false).unwrap());
    output.push_str(&state.process_chunk("e\u{0301}j", false).unwrap());
    output.push_str(&state.process_chunk("a\u{0300} vu!", false).unwrap());
    output.push_str(&state.finalize().unwrap());

    assert_eq!(output, expected);
}