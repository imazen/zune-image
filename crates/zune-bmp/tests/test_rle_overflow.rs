//! Proof of concept for RLE allocation overflow vulnerability
//!
//! This test demonstrates that the RLE allocation at decoder.rs uses
//! unchecked multiplication which can overflow on 32-bit platforms.
//!
//! The vulnerable code at decoder.rs:1142:
//! ```ignore
//! let mut pixels = vec![0; ((self.width * self.height * usize::from(depth)) + 7) >> 3];
//! ```

/// Test that proves overflow occurs with these dimensions on 32-bit
///
/// On 32-bit: 16384 * 16384 * 8 = 2,147,483,648 which is exactly 2^31
/// This fits in u32 but 16384 * 16384 * 32 = 8,589,934,592 overflows
#[test]
fn test_overflow_would_occur_on_32bit() {
    let width: usize = 16384;
    let height: usize = 16384;
    let depth: usize = 32;

    // Simulate what happens on 32-bit by using u32
    let width_32 = width as u32;
    let height_32 = height as u32;
    let depth_32 = depth as u32;

    // The UNCHECKED calculation (what the vulnerable code does)
    let unchecked_result = width_32.wrapping_mul(height_32).wrapping_mul(depth_32);

    // The CHECKED calculation (what safe code should do)
    let checked_result = width_32
        .checked_mul(height_32)
        .and_then(|v| v.checked_mul(depth_32));

    println!("width={width}, height={height}, depth={depth}");
    println!("Unchecked (wrapping): {unchecked_result}");
    println!("Checked: {:?}", checked_result);

    // Prove overflow occurs
    assert!(checked_result.is_none(), "Should overflow on 32-bit");
    assert_eq!(unchecked_result, 0, "Wrapping multiplication should produce 0");

    // This means: on 32-bit, the allocation would be ((0 + 7) >> 3) = 0 bytes!
    // Then accessing pixels[...] would panic or cause UB
}

/// On the actual platform, verify the fix works
/// Uses 32bpp with RLE to trigger: 16384 * 16384 * 32 = 8GB overflow
#[test]
#[cfg(target_pointer_width = "32")]
fn test_actual_32bit_platform() {
    use zune_bmp::BmpDecoder;
    use zune_core::bytestream::ZCursor;

    // Craft RLE BMP with 32bpp to trigger overflow in decode_rle()
    // The vulnerable line: vec![0; ((width * height * depth) + 7) >> 3]
    // With 32bpp: 16384 * 16384 * 32 = 8,589,934,592 > u32::MAX
    let mut data = Vec::new();
    data.extend_from_slice(b"BM");
    data.extend_from_slice(&0u32.to_le_bytes());      // file size (placeholder)
    data.extend_from_slice(&0u32.to_le_bytes());      // reserved
    let pixel_offset: u32 = 14 + 40;                  // no palette for 32bpp
    data.extend_from_slice(&pixel_offset.to_le_bytes());
    data.extend_from_slice(&40u32.to_le_bytes());     // header size
    data.extend_from_slice(&16384u32.to_le_bytes());  // width - within limits
    data.extend_from_slice(&16384u32.to_le_bytes());  // height - within limits
    data.extend_from_slice(&1u16.to_le_bytes());      // planes
    data.extend_from_slice(&32u16.to_le_bytes());     // 32 bits per pixel!
    data.extend_from_slice(&1u32.to_le_bytes());      // RLE8 compression
    data.extend_from_slice(&0u32.to_le_bytes());      // image size
    data.extend_from_slice(&2835u32.to_le_bytes());   // x ppm
    data.extend_from_slice(&2835u32.to_le_bytes());   // y ppm
    data.extend_from_slice(&0u32.to_le_bytes());      // colors used
    data.extend_from_slice(&0u32.to_le_bytes());      // important colors
    // RLE data - end of bitmap
    data.extend_from_slice(&[0x00, 0x01]);
    let file_size = data.len() as u32;
    data[2..6].copy_from_slice(&file_size.to_le_bytes());

    let cursor = ZCursor::new(&data);
    let mut decoder = BmpDecoder::new(cursor);

    // WITHOUT FIX: decode_rle() allocates ((16384*16384*32+7)>>3) bytes
    //   On 32-bit: overflows to ~0, then pixels[...] panics on OOB
    // WITH FIX: Returns Err(OverFlowOccurred) from checked_mul
    let result = decoder.decode();

    // Must return Err, not panic
    assert!(result.is_err(), "Should return overflow error, not panic");
}
