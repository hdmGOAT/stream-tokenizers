use tokenizers::streaming::BoundedBuffer;

#[test]
fn bounded_buffer_append_and_read_unprocessed() {
    let mut buffer = BoundedBuffer::new(16);
    buffer.append(b"hello").unwrap();

    assert_eq!(buffer.unprocessed(), b"hello");
    assert_eq!(buffer.available(), 11);
}

#[test]
fn bounded_buffer_advance_reduces_unprocessed_view() {
    let mut buffer = BoundedBuffer::new(16);
    buffer.append(b"hello world").unwrap();
    buffer.advance(6);

    assert_eq!(buffer.unprocessed(), b"world");
    assert_eq!(buffer.available(), 11);
}

#[test]
fn bounded_buffer_slides_when_capacity_would_be_exceeded() {
    let mut buffer = BoundedBuffer::new(8);
    buffer.append(b"abcd").unwrap();
    buffer.advance(2);

    buffer.append(b"efgh").unwrap();
    assert_eq!(buffer.unprocessed(), b"cdefgh");
    assert_eq!(buffer.available(), 2);
}

#[test]
fn bounded_buffer_rejects_oversized_append() {
    let mut buffer = BoundedBuffer::new(4);

    let error = buffer.append(b"12345").unwrap_err();
    assert!(error.to_string().contains("capacity"));
}

#[test]
fn bounded_buffer_preserves_utf8_bytes_across_chunk_boundaries() {
    let mut buffer = BoundedBuffer::new(8);

    buffer.append(&[0xE2, 0x82]).unwrap();
    buffer.append(&[0xAC]).unwrap();

    let text = std::str::from_utf8(buffer.unprocessed()).unwrap();
    assert_eq!(text, "€");
}