mod bounded_buffer;
mod normalizer_state;
mod orchestrator;
mod pre_tokenizer_state;
mod word_level_streamer;

pub use bounded_buffer::BoundedBuffer;
pub use normalizer_state::NormalizerState;
pub use orchestrator::StreamTokenizer;
pub use pre_tokenizer_state::PreTokenizerState;
pub use word_level_streamer::WordLevelStreamer;

use crate::{Split, Token};
use crate::tokenizer::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingConfig {
	pub requires_word_boundaries: bool,
	pub lookahead_bytes: usize,
	pub can_emit_incrementally: bool,
	pub min_chunk_size: usize,
}

pub trait StreamingTokenizer {
	fn streaming_config() -> StreamingConfig;
	fn process_splits(&mut self, splits: Vec<Split>, is_final: bool) -> Result<Vec<Token>>;
	fn finalize(&mut self) -> Result<Vec<Token>>;
}
