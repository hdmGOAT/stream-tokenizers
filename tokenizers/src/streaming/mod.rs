mod bounded_buffer;
mod bpe_streamer;
mod normalizer_state;
mod orchestrator;
mod pre_tokenizer_state;
mod unigram_streamer;
mod word_level_streamer;
mod wordpiece_streamer;

pub use bounded_buffer::BoundedBuffer;
pub use bpe_streamer::BPEStreamer;
pub use normalizer_state::NormalizerState;
pub use orchestrator::StreamTokenizer;
pub use pre_tokenizer_state::PreTokenizerState;
pub use unigram_streamer::UnigramStreamer;
pub use word_level_streamer::WordLevelStreamer;
pub use wordpiece_streamer::WordPieceStreamer;

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
