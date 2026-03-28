use crate::models::Model;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokenizers as tk;
use tokenizers::models::ModelWrapper;
use tokenizers::streaming::{
  StreamTokenizer as TkStreamTokenizer,
  StreamingTokenizer as TkStreamingTokenizer,
};

#[napi(object)]
pub struct StreamingToken {
  pub id: u32,
  pub value: String,
  pub offsets: Vec<u32>,
}

#[napi(object)]
pub struct StreamingConfig {
  pub requires_word_boundaries: bool,
  pub lookahead_bytes: u32,
  pub can_emit_incrementally: bool,
  pub min_chunk_size: u32,
}

#[napi]
pub struct StreamingTokenizer {
  inner: StreamingTokenizerInner,
}

enum StreamingTokenizerInner {
  WordLevel(TkStreamTokenizer<tk::streaming::WordLevelStreamer>),
  BPE(TkStreamTokenizer<tk::streaming::BPEStreamer>),
  WordPiece(TkStreamTokenizer<tk::streaming::WordPieceStreamer>),
  Unigram(TkStreamTokenizer<tk::streaming::UnigramStreamer>),
}

#[napi]
impl StreamingTokenizer {
  #[napi(constructor)]
  pub fn new(model: &Model, buffer_size: Option<u32>) -> Result<Self> {
    let buffer_size = buffer_size.unwrap_or(1024 * 1024) as usize;
    let model_guard = model
      .model
      .as_ref()
      .ok_or_else(|| Error::from_reason("Uninitialized Model".to_string()))?
      .read()
      .unwrap();

    let inner = match &*model_guard {
      ModelWrapper::WordLevel(wl) => {
        let streamer = tk::streaming::WordLevelStreamer::new(wl.clone());
        StreamingTokenizerInner::WordLevel(TkStreamTokenizer::new(streamer, buffer_size))
      }
      ModelWrapper::BPE(bpe) => {
        let streamer = tk::streaming::BPEStreamer::new(bpe.clone());
        StreamingTokenizerInner::BPE(TkStreamTokenizer::new(streamer, buffer_size))
      }
      ModelWrapper::WordPiece(wp) => {
        let streamer = tk::streaming::WordPieceStreamer::new(wp.clone());
        StreamingTokenizerInner::WordPiece(TkStreamTokenizer::new(streamer, buffer_size))
      }
      ModelWrapper::Unigram(unigram) => {
        let streamer = tk::streaming::UnigramStreamer::new(unigram.clone());
        StreamingTokenizerInner::Unigram(TkStreamTokenizer::new(streamer, buffer_size))
      }
    };

    Ok(Self { inner })
  }

  #[napi]
  pub fn process_chunk(&mut self, chunk: Buffer) -> Result<()> {
    match &mut self.inner {
      StreamingTokenizerInner::WordLevel(st) => st
        .process_chunk(chunk.as_ref())
        .map_err(|e| Error::from_reason(e.to_string()))?,
      StreamingTokenizerInner::BPE(st) => st
        .process_chunk(chunk.as_ref())
        .map_err(|e| Error::from_reason(e.to_string()))?,
      StreamingTokenizerInner::WordPiece(st) => st
        .process_chunk(chunk.as_ref())
        .map_err(|e| Error::from_reason(e.to_string()))?,
      StreamingTokenizerInner::Unigram(st) => st
        .process_chunk(chunk.as_ref())
        .map_err(|e| Error::from_reason(e.to_string()))?,
    }
    Ok(())
  }

  #[napi]
  pub fn finalize(&mut self) -> Result<()> {
    match &mut self.inner {
      StreamingTokenizerInner::WordLevel(st) => {
        st.finalize().map_err(|e| Error::from_reason(e.to_string()))?
      }
      StreamingTokenizerInner::BPE(st) => {
        st.finalize().map_err(|e| Error::from_reason(e.to_string()))?
      }
      StreamingTokenizerInner::WordPiece(st) => {
        st.finalize().map_err(|e| Error::from_reason(e.to_string()))?
      }
      StreamingTokenizerInner::Unigram(st) => {
        st.finalize().map_err(|e| Error::from_reason(e.to_string()))?
      }
    }
    Ok(())
  }

  #[napi]
  pub fn drain_tokens(&mut self) -> Vec<StreamingToken> {
    let tokens = match &mut self.inner {
      StreamingTokenizerInner::WordLevel(st) => st.drain_tokens(),
      StreamingTokenizerInner::BPE(st) => st.drain_tokens(),
      StreamingTokenizerInner::WordPiece(st) => st.drain_tokens(),
      StreamingTokenizerInner::Unigram(st) => st.drain_tokens(),
    };

    tokens
      .into_iter()
      .map(|token| StreamingToken {
        id: token.id,
        value: token.value,
        offsets: vec![token.offsets.0 as u32, token.offsets.1 as u32],
      })
      .collect()
  }

  #[napi]
  pub fn config(&self) -> StreamingConfig {
    let config = match &self.inner {
      StreamingTokenizerInner::WordLevel(_) => tk::streaming::WordLevelStreamer::streaming_config(),
      StreamingTokenizerInner::BPE(_) => tk::streaming::BPEStreamer::streaming_config(),
      StreamingTokenizerInner::WordPiece(_) => tk::streaming::WordPieceStreamer::streaming_config(),
      StreamingTokenizerInner::Unigram(_) => tk::streaming::UnigramStreamer::streaming_config(),
    };

    StreamingConfig {
      requires_word_boundaries: config.requires_word_boundaries,
      lookahead_bytes: config.lookahead_bytes as u32,
      can_emit_incrementally: config.can_emit_incrementally,
      min_chunk_size: config.min_chunk_size as u32,
    }
  }
}
