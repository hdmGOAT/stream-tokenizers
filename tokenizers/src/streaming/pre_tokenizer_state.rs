use unicode_categories::UnicodeCategories;

use crate::tokenizer::Result;

#[derive(Debug, Default, Clone)]
pub struct PreTokenizerState {
    pending: String,
}

impl PreTokenizerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_chunk(&mut self, normalized: &str, is_final: bool) -> Result<Vec<String>> {
        self.pending.push_str(normalized);

        let emit_end = if is_final {
            self.pending.len()
        } else {
            find_last_boundary_end(&self.pending)
        };

        if emit_end == 0 {
            return Ok(vec![]);
        }

        let emit = self.pending[..emit_end].to_string();
        self.pending = self.pending[emit_end..].to_string();

        Ok(tokenize_segment(&emit))
    }

    pub fn finalize(&mut self) -> Result<Vec<String>> {
        self.process_chunk("", true)
    }
}

fn find_last_boundary_end(value: &str) -> usize {
    for (idx, ch) in value.char_indices().rev() {
        if ch.is_whitespace() || is_punctuation(ch) {
            return idx + ch.len_utf8();
        }
    }
    0
}

fn tokenize_segment(segment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in segment.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            continue;
        }

        if is_punctuation(ch) {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
            continue;
        }

        current.push(ch);
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation() || ch.is_punctuation()
}
