use std::collections::VecDeque;

use crate::tokenizer::Result;
use crate::{NormalizedString, Split, Token};

use super::{BoundedBuffer, NormalizerState, PreTokenizerState, StreamingConfig, StreamingTokenizer};

#[derive(Debug, Clone)]
pub struct StreamTokenizer<M: StreamingTokenizer> {
    buffer: BoundedBuffer,
    normalizer_state: NormalizerState,
    pre_tokenizer_state: PreTokenizerState,
    model_state: M,
    config: StreamingConfig,
    completed_tokens: VecDeque<Token>,
    finalized: bool,
}

impl<M: StreamingTokenizer> StreamTokenizer<M> {
    pub fn new(model_state: M, max_buffer_size: usize) -> Self {
        Self {
            buffer: BoundedBuffer::new(max_buffer_size),
            normalizer_state: NormalizerState::new(),
            pre_tokenizer_state: PreTokenizerState::new(),
            model_state,
            config: M::streaming_config(),
            completed_tokens: VecDeque::new(),
            finalized: false,
        }
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    pub fn process_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            return Err("cannot process chunk after finalize".into());
        }

        self.buffer.append(chunk)?;

        let text = std::str::from_utf8(self.buffer.unprocessed())?;
        let normalized = self.normalizer_state.process_chunk(text, false)?;
        let pieces = self.pre_tokenizer_state.process_chunk(&normalized, false)?;
        let splits: Vec<Split> = pieces
            .into_iter()
            .map(|piece| Split::from(NormalizedString::from(piece)))
            .collect();

        if !splits.is_empty() {
            let tokens = self.model_state.process_splits(splits, false)?;
            self.completed_tokens.extend(tokens);
        }

        let consumed = self.buffer.unprocessed().len();
        self.buffer.advance(consumed);
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }

        let text = std::str::from_utf8(self.buffer.unprocessed())?;
        let normalized = self.normalizer_state.process_chunk(text, true)?;
        let pieces = self.pre_tokenizer_state.process_chunk(&normalized, true)?;
        let splits: Vec<Split> = pieces
            .into_iter()
            .map(|piece| Split::from(NormalizedString::from(piece)))
            .collect();

        if !splits.is_empty() {
            let final_chunk_tokens = self.model_state.process_splits(splits, true)?;
            self.completed_tokens.extend(final_chunk_tokens);
        }

        let consumed = self.buffer.unprocessed().len();
        self.buffer.advance(consumed);

        self.finalized = true;
        let tokens = self.model_state.finalize()?;
        self.completed_tokens.extend(tokens);
        Ok(())
    }

    pub fn next_token(&mut self) -> Option<Token> {
        self.completed_tokens.pop_front()
    }

    pub fn drain_tokens(&mut self) -> Vec<Token> {
        self.completed_tokens.drain(..).collect()
    }
}
