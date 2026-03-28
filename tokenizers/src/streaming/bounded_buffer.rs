use crate::tokenizer::Result;

#[derive(Debug, Clone)]
pub struct BoundedBuffer {
    data: Vec<u8>,
    read_pos: usize,
    write_pos: usize,
    max_size: usize,
}

impl BoundedBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: vec![0; max_size],
            read_pos: 0,
            write_pos: 0,
            max_size,
        }
    }

    pub fn append(&mut self, chunk: &[u8]) -> Result<()> {
        if chunk.len() > self.max_size {
            return Err(format!(
                "chunk length {} exceeds buffer capacity {}",
                chunk.len(),
                self.max_size
            )
            .into());
        }

        let unprocessed_len = self.write_pos.saturating_sub(self.read_pos);
        if unprocessed_len + chunk.len() > self.max_size {
            self.slide_window();
        }

        let unprocessed_len = self.write_pos.saturating_sub(self.read_pos);
        if unprocessed_len + chunk.len() > self.max_size {
            return Err(format!(
                "not enough capacity for append: required {}, available {}",
                chunk.len(),
                self.available()
            )
            .into());
        }

        let end = self.write_pos + chunk.len();
        self.data[self.write_pos..end].copy_from_slice(chunk);
        self.write_pos = end;
        Ok(())
    }

    pub fn unprocessed(&self) -> &[u8] {
        &self.data[self.read_pos..self.write_pos]
    }

    pub fn advance(&mut self, bytes: usize) {
        let unprocessed_len = self.write_pos.saturating_sub(self.read_pos);
        let bytes_to_advance = bytes.min(unprocessed_len);
        self.read_pos += bytes_to_advance;

        if self.read_pos == self.write_pos {
            self.read_pos = 0;
            self.write_pos = 0;
        }
    }

    pub fn available(&self) -> usize {
        self.max_size
            .saturating_sub(self.write_pos.saturating_sub(self.read_pos))
    }

    fn slide_window(&mut self) {
        if self.read_pos == 0 {
            return;
        }

        self.data.copy_within(self.read_pos..self.write_pos, 0);
        self.write_pos -= self.read_pos;
        self.read_pos = 0;
    }
}
