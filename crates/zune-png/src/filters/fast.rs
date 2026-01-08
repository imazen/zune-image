/*
 * Copyright (c) 2023.
 *
 * This software is free software; You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Fast auto-vectorizing filter implementations.
//!
//! These implementations use patterns that auto-vectorize well on modern compilers,
//! following the approach proven effective in the `png` crate. The multiversion
//! attribute enables runtime dispatch to optimized versions for different CPU features.

use core::convert::TryInto;
use multiversion::multiversion;

/// Paeth predictor - stb-style formulation that generates branch-free code.
#[inline(always)]
fn paeth_predict(a: u8, b: u8, c: u8) -> u8 {
    let a = i16::from(a);
    let b = i16::from(b);
    let c = i16::from(c);
    let thresh = c * 3 - (a + b);
    let lo = a.min(b);
    let hi = a.max(b);
    let t0 = if hi <= thresh { lo } else { c };
    let t1 = if thresh <= lo { hi } else { t0 };
    t1 as u8
}

/// Truncating average without overflow - from Stanford bit-hacks.
#[inline(always)]
fn avg(a: u8, b: u8) -> u8 {
    // This computes floor((a + b) / 2) without overflow
    (a & b) + ((a ^ b) >> 1)
}

// ============================================================================
// SUB FILTER
// ============================================================================

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_sub_1bpp(current: &mut [u8]) {
    current.iter_mut().reduce(|prev, curr| {
        *curr = curr.wrapping_add(*prev);
        curr
    });
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_sub_2bpp(current: &mut [u8]) {
    let mut prev = [0u8; 2];
    for chunk in current.chunks_exact_mut(2) {
        let new_chunk = [
            chunk[0].wrapping_add(prev[0]),
            chunk[1].wrapping_add(prev[1]),
        ];
        *TryInto::<&mut [u8; 2]>::try_into(chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_sub_3bpp(current: &mut [u8]) {
    let mut prev = [0u8; 3];
    for chunk in current.chunks_exact_mut(3) {
        let new_chunk = [
            chunk[0].wrapping_add(prev[0]),
            chunk[1].wrapping_add(prev[1]),
            chunk[2].wrapping_add(prev[2]),
        ];
        *TryInto::<&mut [u8; 3]>::try_into(chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_sub_4bpp(current: &mut [u8]) {
    let mut prev = [0u8; 4];
    for chunk in current.chunks_exact_mut(4) {
        let new_chunk = [
            chunk[0].wrapping_add(prev[0]),
            chunk[1].wrapping_add(prev[1]),
            chunk[2].wrapping_add(prev[2]),
            chunk[3].wrapping_add(prev[3]),
        ];
        *TryInto::<&mut [u8; 4]>::try_into(chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_sub_6bpp(current: &mut [u8]) {
    let mut prev = [0u8; 6];
    for chunk in current.chunks_exact_mut(6) {
        let new_chunk = [
            chunk[0].wrapping_add(prev[0]),
            chunk[1].wrapping_add(prev[1]),
            chunk[2].wrapping_add(prev[2]),
            chunk[3].wrapping_add(prev[3]),
            chunk[4].wrapping_add(prev[4]),
            chunk[5].wrapping_add(prev[5]),
        ];
        *TryInto::<&mut [u8; 6]>::try_into(chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_sub_8bpp(current: &mut [u8]) {
    let mut prev = [0u8; 8];
    for chunk in current.chunks_exact_mut(8) {
        let new_chunk = [
            chunk[0].wrapping_add(prev[0]),
            chunk[1].wrapping_add(prev[1]),
            chunk[2].wrapping_add(prev[2]),
            chunk[3].wrapping_add(prev[3]),
            chunk[4].wrapping_add(prev[4]),
            chunk[5].wrapping_add(prev[5]),
            chunk[6].wrapping_add(prev[6]),
            chunk[7].wrapping_add(prev[7]),
        ];
        *TryInto::<&mut [u8; 8]>::try_into(chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

/// Generic sub filter for any bpp - copies raw to current first, then adds previous
pub fn unfilter_sub(raw: &[u8], current: &mut [u8], bpp: usize) {
    // Copy raw to current
    let len = raw.len().min(current.len());
    current[..len].copy_from_slice(&raw[..len]);

    // Dispatch to optimized version based on bpp
    match bpp {
        1 => unfilter_sub_1bpp(current),
        2 => unfilter_sub_2bpp(current),
        3 => unfilter_sub_3bpp(current),
        4 => unfilter_sub_4bpp(current),
        6 => unfilter_sub_6bpp(current),
        8 => unfilter_sub_8bpp(current),
        _ => unfilter_sub_generic(current, bpp),
    }
}

fn unfilter_sub_generic(current: &mut [u8], bpp: usize) {
    for i in bpp..current.len() {
        current[i] = current[i].wrapping_add(current[i - bpp]);
    }
}

// ============================================================================
// UP FILTER
// ============================================================================

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_up(previous: &[u8], current: &mut [u8]) {
    // Process in chunks of 32 bytes for better vectorization
    let len = previous.len().min(current.len());
    let (prev_chunks, prev_remainder) = previous[..len].as_chunks::<32>();
    let (cur_chunks, cur_remainder) = current[..len].as_chunks_mut::<32>();

    for (prev, cur) in prev_chunks.iter().zip(cur_chunks.iter_mut()) {
        for i in 0..32 {
            cur[i] = cur[i].wrapping_add(prev[i]);
        }
    }

    for (p, c) in prev_remainder.iter().zip(cur_remainder.iter_mut()) {
        *c = c.wrapping_add(*p);
    }
}

// ============================================================================
// AVG FILTER
// ============================================================================

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_avg_3bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 3 {
        return;
    }

    // First pixel: avg(0, b)
    current[0] = current[0].wrapping_add(previous[0] >> 1);
    current[1] = current[1].wrapping_add(previous[1] >> 1);
    current[2] = current[2].wrapping_add(previous[2] >> 1);

    let mut prev = [current[0], current[1], current[2]];

    for (prev_chunk, cur_chunk) in previous[3..len].chunks_exact(3).zip(current[3..len].chunks_exact_mut(3)) {
        let new_chunk = [
            cur_chunk[0].wrapping_add(avg(prev[0], prev_chunk[0])),
            cur_chunk[1].wrapping_add(avg(prev[1], prev_chunk[1])),
            cur_chunk[2].wrapping_add(avg(prev[2], prev_chunk[2])),
        ];
        *TryInto::<&mut [u8; 3]>::try_into(cur_chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_avg_4bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 4 {
        return;
    }

    // First pixel: avg(0, b)
    current[0] = current[0].wrapping_add(previous[0] >> 1);
    current[1] = current[1].wrapping_add(previous[1] >> 1);
    current[2] = current[2].wrapping_add(previous[2] >> 1);
    current[3] = current[3].wrapping_add(previous[3] >> 1);

    let mut prev = [current[0], current[1], current[2], current[3]];

    for (prev_chunk, cur_chunk) in previous[4..len].chunks_exact(4).zip(current[4..len].chunks_exact_mut(4)) {
        let new_chunk = [
            cur_chunk[0].wrapping_add(avg(prev[0], prev_chunk[0])),
            cur_chunk[1].wrapping_add(avg(prev[1], prev_chunk[1])),
            cur_chunk[2].wrapping_add(avg(prev[2], prev_chunk[2])),
            cur_chunk[3].wrapping_add(avg(prev[3], prev_chunk[3])),
        ];
        *TryInto::<&mut [u8; 4]>::try_into(cur_chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_avg_6bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 6 {
        return;
    }

    // First pixel: avg(0, b)
    for i in 0..6 {
        current[i] = current[i].wrapping_add(previous[i] >> 1);
    }

    let mut prev = [current[0], current[1], current[2], current[3], current[4], current[5]];

    for (prev_chunk, cur_chunk) in previous[6..len].chunks_exact(6).zip(current[6..len].chunks_exact_mut(6)) {
        let new_chunk = [
            cur_chunk[0].wrapping_add(avg(prev[0], prev_chunk[0])),
            cur_chunk[1].wrapping_add(avg(prev[1], prev_chunk[1])),
            cur_chunk[2].wrapping_add(avg(prev[2], prev_chunk[2])),
            cur_chunk[3].wrapping_add(avg(prev[3], prev_chunk[3])),
            cur_chunk[4].wrapping_add(avg(prev[4], prev_chunk[4])),
            cur_chunk[5].wrapping_add(avg(prev[5], prev_chunk[5])),
        ];
        *TryInto::<&mut [u8; 6]>::try_into(cur_chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_avg_8bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 8 {
        return;
    }

    // First pixel: avg(0, b)
    for i in 0..8 {
        current[i] = current[i].wrapping_add(previous[i] >> 1);
    }

    let mut prev = [
        current[0], current[1], current[2], current[3],
        current[4], current[5], current[6], current[7],
    ];

    for (prev_chunk, cur_chunk) in previous[8..len].chunks_exact(8).zip(current[8..len].chunks_exact_mut(8)) {
        let new_chunk = [
            cur_chunk[0].wrapping_add(avg(prev[0], prev_chunk[0])),
            cur_chunk[1].wrapping_add(avg(prev[1], prev_chunk[1])),
            cur_chunk[2].wrapping_add(avg(prev[2], prev_chunk[2])),
            cur_chunk[3].wrapping_add(avg(prev[3], prev_chunk[3])),
            cur_chunk[4].wrapping_add(avg(prev[4], prev_chunk[4])),
            cur_chunk[5].wrapping_add(avg(prev[5], prev_chunk[5])),
            cur_chunk[6].wrapping_add(avg(prev[6], prev_chunk[6])),
            cur_chunk[7].wrapping_add(avg(prev[7], prev_chunk[7])),
        ];
        *TryInto::<&mut [u8; 8]>::try_into(cur_chunk).unwrap() = new_chunk;
        prev = new_chunk;
    }
}

/// Generic avg filter dispatcher
pub fn unfilter_avg(previous: &[u8], raw: &[u8], current: &mut [u8], bpp: usize) {
    let len = raw.len().min(current.len());
    current[..len].copy_from_slice(&raw[..len]);

    match bpp {
        3 => unfilter_avg_3bpp(previous, current),
        4 => unfilter_avg_4bpp(previous, current),
        6 => unfilter_avg_6bpp(previous, current),
        8 => unfilter_avg_8bpp(previous, current),
        _ => unfilter_avg_generic(previous, current, bpp),
    }
}

fn unfilter_avg_generic(previous: &[u8], current: &mut [u8], bpp: usize) {
    let len = previous.len().min(current.len());

    // First bpp bytes: avg(0, b)
    for i in 0..bpp.min(len) {
        current[i] = current[i].wrapping_add(previous[i] >> 1);
    }

    // Rest: avg(a, b)
    for i in bpp..len {
        current[i] = current[i].wrapping_add(avg(current[i - bpp], previous[i]));
    }
}

// ============================================================================
// PAETH FILTER
// ============================================================================

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_paeth_3bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 3 {
        return;
    }

    // First pixel: paeth(0, b, 0) = b
    current[0] = current[0].wrapping_add(previous[0]);
    current[1] = current[1].wrapping_add(previous[1]);
    current[2] = current[2].wrapping_add(previous[2]);

    // Process remaining pixels byte by byte (sequential dependency prevents vectorization)
    for i in 3..len {
        let a = current[i - 3];
        let b = previous[i];
        let c = previous[i - 3];
        current[i] = current[i].wrapping_add(paeth_predict(a, b, c));
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_paeth_4bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 4 {
        return;
    }

    // First pixel: paeth(0, b, 0) = b
    current[0] = current[0].wrapping_add(previous[0]);
    current[1] = current[1].wrapping_add(previous[1]);
    current[2] = current[2].wrapping_add(previous[2]);
    current[3] = current[3].wrapping_add(previous[3]);

    // Process remaining pixels byte by byte (sequential dependency prevents vectorization)
    for i in 4..len {
        let a = current[i - 4];
        let b = previous[i];
        let c = previous[i - 4];
        current[i] = current[i].wrapping_add(paeth_predict(a, b, c));
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_paeth_6bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 6 {
        return;
    }

    // First pixel: paeth(0, b, 0) = b
    for i in 0..6 {
        current[i] = current[i].wrapping_add(previous[i]);
    }

    // Process remaining pixels byte by byte
    for i in 6..len {
        let a = current[i - 6];
        let b = previous[i];
        let c = previous[i - 6];
        current[i] = current[i].wrapping_add(paeth_predict(a, b, c));
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
pub fn unfilter_paeth_8bpp(previous: &[u8], current: &mut [u8]) {
    let len = previous.len().min(current.len());
    if len < 8 {
        return;
    }

    // First pixel: paeth(0, b, 0) = b
    for i in 0..8 {
        current[i] = current[i].wrapping_add(previous[i]);
    }

    // Process remaining pixels byte by byte
    for i in 8..len {
        let a = current[i - 8];
        let b = previous[i];
        let c = previous[i - 8];
        current[i] = current[i].wrapping_add(paeth_predict(a, b, c));
    }
}

/// Generic paeth filter dispatcher
pub fn unfilter_paeth(previous: &[u8], raw: &[u8], current: &mut [u8], bpp: usize) {
    let len = raw.len().min(current.len());
    current[..len].copy_from_slice(&raw[..len]);

    match bpp {
        3 => unfilter_paeth_3bpp(previous, current),
        4 => unfilter_paeth_4bpp(previous, current),
        6 => unfilter_paeth_6bpp(previous, current),
        8 => unfilter_paeth_8bpp(previous, current),
        _ => unfilter_paeth_generic(previous, current, bpp),
    }
}

fn unfilter_paeth_generic(previous: &[u8], current: &mut [u8], bpp: usize) {
    let len = previous.len().min(current.len());

    // First bpp bytes: paeth(0, b, 0) = b
    for i in 0..bpp.min(len) {
        current[i] = current[i].wrapping_add(previous[i]);
    }

    // Rest
    for i in bpp..len {
        let a = current[i - bpp];
        let b = previous[i];
        let c = previous[i - bpp];
        current[i] = current[i].wrapping_add(paeth_predict(a, b, c));
    }
}

// ============================================================================
// FIRST ROW VARIANTS (no previous row)
// ============================================================================

/// Avg filter for first row (previous row is all zeros)
pub fn unfilter_avg_first(raw: &[u8], current: &mut [u8], bpp: usize) {
    let len = raw.len().min(current.len());
    current[..len].copy_from_slice(&raw[..len]);

    // First bpp bytes stay as-is (avg(0, 0) = 0)
    // Rest: avg(a, 0) = a >> 1
    for i in bpp..len {
        current[i] = current[i].wrapping_add(current[i - bpp] >> 1);
    }
}

/// Paeth filter for first row (previous row is all zeros) - equivalent to Sub
pub fn unfilter_paeth_first(raw: &[u8], current: &mut [u8], bpp: usize) {
    // When previous row is all zeros, paeth(a, 0, 0) = a
    unfilter_sub(raw, current, bpp);
}
