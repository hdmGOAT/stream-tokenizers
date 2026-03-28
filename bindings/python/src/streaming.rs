use pyo3::prelude::*;
use pyo3::types::*;
use tk::streaming::{StreamTokenizer as TkStreamTokenizer, StreamingTokenizer};
use tk::tokenizer::ModelWrapper;
use tokenizers as tk;

use super::models::PyModel;
use super::error::ToPyResult;

/// A streaming tokenizer that processes input in chunks and emits tokens incrementally.
///
/// This is useful for processing large inputs with bounded memory usage.
///
/// Args:
///     model (:class:`~tokenizers.Model`): The model to use for tokenization
///     buffer_size (:obj:`int`, defaults to 1048576): Maximum buffer size in bytes
///
/// Example:
///     ```python
///     from tokenizers import Tokenizer
///     from tokenizers.streaming import StreamingTokenizer
///
///     # Load a tokenizer
///     tokenizer = Tokenizer.from_file("tokenizer.json")
///
///     # Create a streaming tokenizer
///     stream = StreamingTokenizer(tokenizer.model, buffer_size=1024*1024)
///
///     # Process input in chunks
///     stream.process_chunk(b"Hello ")
///     stream.process_chunk(b"world!")
///     stream.finalize()
///
///     # Get all emitted tokens
///     tokens = stream.drain_tokens()
///     ```
#[pyclass(module = "tokenizers.streaming", name = "StreamingTokenizer")]
pub struct PyStreamingTokenizer {
    // We store an enum to support different model types
    inner: StreamingTokenizerInner,
}

enum StreamingTokenizerInner {
    WordLevel(TkStreamTokenizer<tk::streaming::WordLevelStreamer>),
    BPE(TkStreamTokenizer<tk::streaming::BPEStreamer>),
    WordPiece(TkStreamTokenizer<tk::streaming::WordPieceStreamer>),
    Unigram(TkStreamTokenizer<tk::streaming::UnigramStreamer>),
}

#[pymethods]
impl PyStreamingTokenizer {
    /// Create a new streaming tokenizer from a PyModel
    #[new]
    #[pyo3(signature = (model, buffer_size=1048576))]
    fn new(model: &PyModel, buffer_size: usize) -> PyResult<Self> {
        let model_guard = model.model.read().unwrap();
        
        match &*model_guard {
            ModelWrapper::WordLevel(wl) => {
                let streamer = tk::streaming::WordLevelStreamer::new(wl.clone());
                Ok(PyStreamingTokenizer {
                    inner: StreamingTokenizerInner::WordLevel(
                        TkStreamTokenizer::new(streamer, buffer_size),
                    ),
                })
            }
            ModelWrapper::BPE(bpe) => {
                let streamer = tk::streaming::BPEStreamer::new(bpe.clone());
                Ok(PyStreamingTokenizer {
                    inner: StreamingTokenizerInner::BPE(
                        TkStreamTokenizer::new(streamer, buffer_size),
                    ),
                })
            }
            ModelWrapper::WordPiece(wp) => {
                let streamer = tk::streaming::WordPieceStreamer::new(wp.clone());
                Ok(PyStreamingTokenizer {
                    inner: StreamingTokenizerInner::WordPiece(
                        TkStreamTokenizer::new(streamer, buffer_size),
                    ),
                })
            }
            ModelWrapper::Unigram(unigram) => {
                let streamer = tk::streaming::UnigramStreamer::new(unigram.clone());
                Ok(PyStreamingTokenizer {
                    inner: StreamingTokenizerInner::Unigram(
                        TkStreamTokenizer::new(streamer, buffer_size),
                    ),
                })
            }
        }
    }

    /// Process a chunk of input bytes
    ///
    /// Args:
    ///     chunk (:obj:`bytes`): The chunk to process
    fn process_chunk(&mut self, chunk: &[u8]) -> PyResult<()> {
        match &mut self.inner {
            StreamingTokenizerInner::WordLevel(st) => {
                ToPyResult(st.process_chunk(chunk)).into_py()?;
            }
            StreamingTokenizerInner::BPE(st) => {
                ToPyResult(st.process_chunk(chunk)).into_py()?;
            }
            StreamingTokenizerInner::WordPiece(st) => {
                ToPyResult(st.process_chunk(chunk)).into_py()?;
            }
            StreamingTokenizerInner::Unigram(st) => {
                ToPyResult(st.process_chunk(chunk)).into_py()?;
            }
        }
        Ok(())
    }

    /// Finalize processing and emit any remaining tokens
    fn finalize(&mut self) -> PyResult<()> {
        match &mut self.inner {
            StreamingTokenizerInner::WordLevel(st) => {
                ToPyResult(st.finalize()).into_py()?;
            }
            StreamingTokenizerInner::BPE(st) => {
                ToPyResult(st.finalize()).into_py()?;
            }
            StreamingTokenizerInner::WordPiece(st) => {
                ToPyResult(st.finalize()).into_py()?;
            }
            StreamingTokenizerInner::Unigram(st) => {
                ToPyResult(st.finalize()).into_py()?;
            }
        }
        Ok(())
    }

    /// Get and remove all emitted tokens
    ///
    /// Returns:
    ///     :obj:`list` of :obj:`dict`: List of token dicts with 'id', 'value', and 'offsets' keys
    fn drain_tokens(&mut self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let tokens = match &mut self.inner {
            StreamingTokenizerInner::WordLevel(st) => st.drain_tokens(),
            StreamingTokenizerInner::BPE(st) => st.drain_tokens(),
            StreamingTokenizerInner::WordPiece(st) => st.drain_tokens(),
            StreamingTokenizerInner::Unigram(st) => st.drain_tokens(),
        };

        let py_tokens = PyList::empty(py);
        for token in tokens {
            let py_token = PyDict::new(py);
            py_token.set_item("id", token.id)?;
            py_token.set_item("value", token.value)?;
            py_token.set_item("offsets", (token.offsets.0, token.offsets.1))?;
            py_tokens.append(py_token)?;
        }
        Ok(py_tokens.into())
    }

    /// Get the streaming configuration
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let config = match &self.inner {
            StreamingTokenizerInner::WordLevel(_) => tk::streaming::WordLevelStreamer::streaming_config(),
            StreamingTokenizerInner::BPE(_) => tk::streaming::BPEStreamer::streaming_config(),
            StreamingTokenizerInner::WordPiece(_) => tk::streaming::WordPieceStreamer::streaming_config(),
            StreamingTokenizerInner::Unigram(_) => tk::streaming::UnigramStreamer::streaming_config(),
        };

        let py_config = PyDict::new(py);
        py_config.set_item("requires_word_boundaries", config.requires_word_boundaries)?;
        py_config.set_item("lookahead_bytes", config.lookahead_bytes)?;
        py_config.set_item("can_emit_incrementally", config.can_emit_incrementally)?;
        py_config.set_item("min_chunk_size", config.min_chunk_size)?;
        Ok(py_config.into())
    }
}


