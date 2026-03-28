use crate::models::wordlevel::WordLevel;
use crate::tokenizer::{Model, Result};
use crate::{Split, Token};

use super::{StreamingConfig, StreamingTokenizer};

#[derive(Debug, Clone)]
pub struct WordLevelStreamer {
    model: WordLevel,
}

impl WordLevelStreamer {
    pub fn new(model: WordLevel) -> Self {
        Self { model }
    }
}

impl StreamingTokenizer for WordLevelStreamer {
    fn streaming_config() -> StreamingConfig {
        StreamingConfig {
            requires_word_boundaries: true,
            lookahead_bytes: 0,
            can_emit_incrementally: true,
            min_chunk_size: 1,
        }
    }

    fn process_splits(&mut self, splits: Vec<Split>, _is_final: bool) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        for split in splits {
            tokens.extend(self.model.tokenize(split.text())?);
        }

        Ok(tokens)
    }

    fn finalize(&mut self) -> Result<Vec<Token>> {
        Ok(vec![])
    }
}
