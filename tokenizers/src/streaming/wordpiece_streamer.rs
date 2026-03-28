use crate::models::wordpiece::WordPiece;
use crate::tokenizer::{Model, Result};
use crate::{Split, Token};

use super::{StreamingConfig, StreamingTokenizer};

#[derive(Debug, Clone)]
pub struct WordPieceStreamer {
    model: WordPiece,
}

impl WordPieceStreamer {
    pub fn new(model: WordPiece) -> Self {
        Self { model }
    }
}

impl StreamingTokenizer for WordPieceStreamer {
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
