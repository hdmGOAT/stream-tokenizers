use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

use crate::tokenizer::Result;

#[derive(Debug, Default, Clone)]
pub struct NormalizerState {
    pending: String,
}

impl NormalizerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_chunk(&mut self, chunk: &str, is_final: bool) -> Result<String> {
        self.pending.push_str(chunk);

        if self.pending.is_empty() {
            return Ok(String::new());
        }

        if is_final {
            return self.finalize();
        }

        let emit_end = self.safe_emit_end();
        if emit_end == 0 {
            return Ok(String::new());
        }

        let emitted = self.pending[..emit_end].to_string();
        self.pending = self.pending[emit_end..].to_string();

        Ok(normalize_nfc(&emitted))
    }

    pub fn finalize(&mut self) -> Result<String> {
        let pending = std::mem::take(&mut self.pending);
        Ok(normalize_nfc(&pending))
    }

    fn safe_emit_end(&self) -> usize {
        let mut reverse_iter = self.pending.char_indices().rev();

        let Some((last_idx, last_char)) = reverse_iter.next() else {
            return 0;
        };

        let mut retain_start = last_idx;

        if is_combining_mark(last_char) {
            for (idx, ch) in reverse_iter {
                retain_start = idx;
                if !is_combining_mark(ch) {
                    break;
                }
            }
        }

        retain_start
    }
}

fn is_combining_mark(ch: char) -> bool {
    ch.is_mark_nonspacing() || ch.is_mark_spacing_combining() || ch.is_mark_enclosing()
}

fn normalize_nfc(value: &str) -> String {
    value.nfc().map(|(ch, _)| ch).collect::<String>()
}
