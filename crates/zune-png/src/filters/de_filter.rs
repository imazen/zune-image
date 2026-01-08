/*
 * Copyright (c) 2023.
 *
 * This software is free software; You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */
#![allow(dead_code, unused_variables)]

use super::fast;

#[allow(clippy::manual_memcpy)]
pub fn handle_avg(
    prev_row: &[u8], raw: &[u8], current: &mut [u8], components: usize, _use_sse4: bool,
) {
    if raw.len() < components || current.len() < components {
        return;
    }
    fast::unfilter_avg(prev_row, raw, current, components);
}

#[allow(clippy::manual_memcpy)]
pub fn handle_sub(raw: &[u8], current: &mut [u8], components: usize, _use_sse2: bool) {
    if current.len() < components || raw.len() < components {
        return;
    }
    fast::unfilter_sub(raw, current, components);
}

#[allow(clippy::manual_memcpy)]
pub fn handle_paeth(
    prev_row: &[u8], raw: &[u8], current: &mut [u8], components: usize, _use_sse4: bool,
) {
    if raw.len() < components || current.len() < components {
        return;
    }
    fast::unfilter_paeth(prev_row, raw, current, components);
}

pub fn handle_up(prev_row: &[u8], raw: &[u8], current: &mut [u8]) {
    let len = raw.len().min(current.len());
    current[..len].copy_from_slice(&raw[..len]);
    fast::unfilter_up(prev_row, current);
}

/// Handle images with the first scanline as paeth scanline
///
/// Special in that the above row is treated as zero
#[allow(clippy::manual_memcpy)]
pub fn handle_paeth_first(raw: &[u8], current: &mut [u8], components: usize) {
    if raw.len() < components || current.len() < components {
        return;
    }
    fast::unfilter_paeth_first(raw, current, components);
}

/// Handle images with the first scanline as an average scanline
///
/// The above row is treated as zero
#[allow(clippy::manual_memcpy)]
pub fn handle_avg_first(raw: &[u8], current: &mut [u8], components: usize) {
    if raw.len() < components || current.len() < components {
        return;
    }
    fast::unfilter_avg_first(raw, current, components);
}

/// Paeth predictor - exported for compatibility
#[inline(always)]
pub fn paeth(a: u8, b: u8, c: u8) -> u8 {
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
