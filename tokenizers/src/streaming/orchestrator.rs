use std::collections::VecDeque;

use crate::tokenizer::Result;
use crate::{Split, Token};

use super::{BoundedBuffer, StreamingConfig, StreamingTokenizer};

#[derive(Debug, Clone)]
pub struct StreamTokenizer<M: StreamingTokenizer> {
    buffer: BoundedBuffer,
    model_state: M,
    config: StreamingConfig,
    completed_tokens: VecDeque<Token>,
    finalized: bool,
}

impl<M: StreamingTokenizer> StreamTokenizer<M> {
    pub fn new(model_state: M, max_buffer_size: usize) -> Self {
        Self {
            buffer: BoundedBuffer::new(max_buffer_size),
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

        let tokens = self
            .model_state
            .process_splits(Vec::<Split>::new(), false)?;
        self.completed_tokens.extend(tokens);

        let consumed = self.buffer.unprocessed().len();
        self.buffer.advance(consumed);
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }

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
