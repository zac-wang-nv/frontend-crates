// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Container demuxing through FFmpeg's libavformat.
//!
//! The crate links the libavformat Dynamo builds in-tree: LGPL-only, with the
//! MOV / MP4 and Matroska / WebM demuxers and no H.264 or HEVC decoder or
//! parser (`scripts/build-ffmpeg.sh` reproduces it). libavformat only reads the
//! container: each packet's bytes go to NVDEC or libvpx unchanged.
//!
//! libavformat reads through a custom I/O context over the caller's
//! [`ByteSource`], with no protocol or file access of its own, and only with
//! the demuxer the container's magic bytes name (no format probing). It
//! applies MP4 edit lists: a packet outside the list's window comes back
//! flagged to discard (decoded, not shown), and times are player times, a
//! leading empty edit included. Keyframe flags come from `stss`, the fragment
//! sample flags or the Matroska block flags, as FFmpeg reads them; `seek`
//! checks each one it starts at in the bitstream.
//!
//! Every packet is read once while demuxing, because only then does
//! libavformat give its presentation time. MP4 packets are then dropped and
//! read again from the source when decoded; WebM / Matroska packets, which
//! libavformat may rewrite (header stripping), are held.

use std::ffi::{CStr, c_int, c_void};
use std::io::{Read, Seek, SeekFrom};

use ffmpeg_sys_next as ff;

use super::source::{ReadFailure, SourceReader};
use super::{
    ByteSource, MAX_PACKETS, Packet, Payload, Track, VideoCodec, Vp9Colour, packet_displays,
    read_error,
};
use crate::{MmError, Result};

/// The container a clip's first bytes name.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Container {
    Mp4,
    Matroska,
}

impl Container {
    fn demuxer(self) -> &'static CStr {
        match self {
            Self::Mp4 => c"mov",
            Self::Matroska => c"matroska",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mp4 => "MP4",
            Self::Matroska => "WebM / Matroska",
        }
    }
}

/// Size of the buffer libavformat reads the source through.
const IO_BUFFER: usize = 64 << 10;

// Linux `errno` and `whence` values (this module is Linux-only).
const EIO: c_int = 5;
const EINVAL: c_int = 22;
const SEEK_SET: c_int = 0;
const SEEK_CUR: c_int = 1;
const SEEK_END: c_int = 2;

/// The opaque state of the custom I/O context.
struct Io<'a> {
    reader: SourceReader<'a>,
    len: u64,
}

unsafe extern "C" fn read_packet(opaque: *mut c_void, buf: *mut u8, size: c_int) -> c_int {
    // SAFETY: `opaque` is the `Io` boxed in `Demuxer::open`, alive while the
    // context is; libavformat passes a buffer of `size` bytes.
    let (io, buf) = unsafe {
        (
            &mut *opaque.cast::<Io<'_>>(),
            std::slice::from_raw_parts_mut(buf, size.max(0) as usize),
        )
    };
    match io.reader.read(buf) {
        Ok(0) => ff::AVERROR_EOF,
        Ok(n) => n as c_int,
        // The reader remembers the failure for `Demuxer::error`.
        Err(_) => ff::AVERROR(EIO),
    }
}

unsafe extern "C" fn seek(opaque: *mut c_void, offset: i64, whence: c_int) -> i64 {
    // SAFETY: as in `read_packet`.
    let io = unsafe { &mut *opaque.cast::<Io<'_>>() };
    if whence & ff::AVSEEK_SIZE != 0 {
        return i64::try_from(io.len).unwrap_or(i64::MAX);
    }
    let to = match whence & !ff::AVSEEK_FORCE {
        SEEK_SET => match u64::try_from(offset) {
            Ok(n) => SeekFrom::Start(n),
            Err(_) => return ff::AVERROR(EINVAL).into(),
        },
        SEEK_CUR => SeekFrom::Current(offset),
        SEEK_END => SeekFrom::End(offset),
        _ => return ff::AVERROR(EINVAL).into(),
    };
    match io.reader.seek(to) {
        Ok(pos) => i64::try_from(pos).unwrap_or(i64::MAX),
        Err(_) => ff::AVERROR(EINVAL).into(),
    }
}

/// An open libavformat context over a [`ByteSource`]; frees everything when
/// dropped.
struct Demuxer<'a> {
    context: *mut ff::AVFormatContext,
    io: *mut ff::AVIOContext,
    opaque: *mut Io<'a>,
    failure: ReadFailure,
    container: Container,
}

impl Drop for Demuxer<'_> {
    fn drop(&mut self) {
        // SAFETY: each pointer was allocated in `open` and is freed once, the
        // format context before the I/O context it reads through. A custom
        // I/O context's buffer may have been replaced by libavformat; free
        // the current one.
        unsafe {
            ff::avformat_close_input(&mut self.context);
            if !self.io.is_null() {
                ff::av_freep((&raw mut (*self.io).buffer).cast());
                ff::avio_context_free(&mut self.io);
            }
            drop(Box::from_raw(self.opaque));
        }
    }
}

impl<'a> Demuxer<'a> {
    fn open(source: &'a dyn ByteSource, container: Container) -> Result<Self> {
        let (reader, failure) = SourceReader::new(source);
        let opaque = Box::into_raw(Box::new(Io {
            reader,
            len: source.len(),
        }));
        let mut demuxer = Self {
            context: std::ptr::null_mut(),
            io: std::ptr::null_mut(),
            opaque,
            failure,
            container,
        };
        let no_memory = || MmError::internal("libavformat could not allocate");
        // SAFETY: the buffer and context are owned by `demuxer` from here, and
        // freed by its `Drop`; the callbacks match libavformat's signatures.
        unsafe {
            let buffer = ff::av_malloc(IO_BUFFER).cast::<u8>();
            if buffer.is_null() {
                return Err(no_memory());
            }
            demuxer.io = ff::avio_alloc_context(
                buffer,
                IO_BUFFER as c_int,
                0,
                opaque.cast(),
                Some(read_packet),
                None,
                Some(seek),
            );
            if demuxer.io.is_null() {
                ff::av_free(buffer.cast());
                return Err(no_memory());
            }
            let context = ff::avformat_alloc_context();
            if context.is_null() {
                return Err(no_memory());
            }
            (*context).pb = demuxer.io;
            (*context).flags |= ff::AVFMT_FLAG_CUSTOM_IO;
            let format = ff::av_find_input_format(container.demuxer().as_ptr());
            if format.is_null() {
                ff::avformat_free_context(context);
                return Err(MmError::unsupported(format!(
                    "the linked FFmpeg has no {} demuxer",
                    container.name()
                )));
            }
            // On failure this frees `context` and sets it to null.
            let mut opened = context;
            let code = ff::avformat_open_input(
                &mut opened,
                std::ptr::null(),
                format,
                std::ptr::null_mut(),
            );
            demuxer.context = opened;
            if code < 0 {
                return Err(demuxer.error(code, "unreadable"));
            }
        }
        Ok(demuxer)
    }

    /// The error for libavformat's `code`: the source's own failure if a read
    /// failed (a short source is bad input, anything else `Internal`), else bad
    /// input.
    fn error(&self, code: c_int, what: &str) -> MmError {
        if let Some(io) = self.failure.take() {
            return read_error(io, &format!("the {} file", self.container.name()));
        }
        MmError::invalid_input(format!(
            "{what} {}: {}",
            self.container.name(),
            av_error(code)
        ))
    }
}

fn av_error(code: c_int) -> String {
    let mut text = [0 as std::ffi::c_char; 128];
    // SAFETY: a valid buffer of the given size, NUL-terminated on return.
    unsafe {
        ff::av_strerror(code, text.as_mut_ptr(), text.len());
        CStr::from_ptr(text.as_ptr()).to_string_lossy().into_owned()
    }
}

fn four_cc(tag: u32) -> [u8; 4] {
    tag.to_le_bytes()
}

/// Read the first video track of `source`, a `container` file, holding at
/// most `max_input_bytes` of WebM / Matroska packets.
pub(super) fn demux(
    source: &dyn ByteSource,
    container: Container,
    max_input_bytes: usize,
) -> Result<Track<'_>> {
    if container == Container::Matroska {
        matroska_segment_fits(source)?;
    }
    let demuxer = Demuxer::open(source, container)?;
    // SAFETY: the context is open; its streams and their parameters are valid
    // until it is closed, and are only read.
    let (index, codec, width, height, config, tag, time_base) = unsafe {
        let context = &*demuxer.context;
        let streams = std::slice::from_raw_parts(context.streams, context.nb_streams as usize);
        let video = streams.iter().map(|&s| &*s).find(|s| {
            (*s.codecpar).codec_type == ff::AVMediaType::AVMEDIA_TYPE_VIDEO
                && s.disposition & ff::AV_DISPOSITION_ATTACHED_PIC == 0
        });
        let Some(video) = video else {
            return Err(MmError::invalid_input(format!(
                "{} file has no video track",
                container.name()
            )));
        };
        // Other tracks' packets are skipped, not read.
        for &stream in streams {
            if (*stream).index != video.index {
                (*stream).discard = ff::AVDiscard::AVDISCARD_ALL;
            }
        }
        let par = &*video.codecpar;
        let codec = match par.codec_id {
            ff::AVCodecID::AV_CODEC_ID_H264 => VideoCodec::H264,
            ff::AVCodecID::AV_CODEC_ID_HEVC => VideoCodec::Hevc,
            ff::AVCodecID::AV_CODEC_ID_VP8 => VideoCodec::Vp8,
            ff::AVCodecID::AV_CODEC_ID_VP9 => VideoCodec::Vp9,
            ff::AVCodecID::AV_CODEC_ID_AV1 => VideoCodec::Av1,
            other => {
                let name = CStr::from_ptr(ff::avcodec_get_name(other)).to_string_lossy();
                return Err(MmError::unsupported(format!(
                    "unsupported {} video codec {name}",
                    container.name()
                )));
            }
        };
        let size = |v: c_int| {
            u32::try_from(v)
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| MmError::invalid_input("video track has an empty frame size"))
        };
        let config = (!par.extradata.is_null() && par.extradata_size > 0).then(|| {
            std::slice::from_raw_parts(par.extradata, par.extradata_size as usize).to_vec()
        });
        (
            video.index,
            codec,
            size(par.width)?,
            size(par.height)?,
            config,
            par.codec_tag,
            video.time_base,
        )
    };
    if time_base.num <= 0 || time_base.den <= 0 {
        return Err(MmError::invalid_input("video track has no time base"));
    }
    let seconds = |t: i64| t as f64 * f64::from(time_base.num) / f64::from(time_base.den);

    let mut packets = Vec::new();
    let mut vp9_colour = Vp9Colour::default();
    let mut offsets = std::collections::HashSet::new();
    // The latest end of a shown frame, when packets say how long they last.
    let mut last_end: Option<f64> = None;
    let mut held: u64 = 0;
    // SAFETY: a packet owned here, freed below.
    let mut packet = unsafe { ff::av_packet_alloc() };
    if packet.is_null() {
        return Err(MmError::internal("libavformat could not allocate"));
    }
    let result = (|| -> Result<()> {
        loop {
            // SAFETY: an open context and an allocated packet, emptied before
            // each read (freeing the packet frees what it holds).
            let code = unsafe {
                ff::av_packet_unref(packet);
                ff::av_read_frame(demuxer.context, packet)
            };
            if code == ff::AVERROR_EOF {
                return Ok(());
            }
            if code < 0 {
                return Err(demuxer.error(code, "unreadable"));
            }
            // SAFETY: as above; the packet's data is valid until it is
            // emptied.
            let p = unsafe { &*packet };
            if p.stream_index != index {
                continue;
            }
            if packets.len() >= MAX_PACKETS {
                return Err(MmError::invalid_input(
                    "video has more packets than it can hold",
                ));
            }
            if p.pts == ff::AV_NOPTS_VALUE || p.data.is_null() || p.size <= 0 {
                return Err(MmError::invalid_input(format!(
                    "{} packet without a time or data",
                    container.name()
                )));
            }
            // SAFETY: libavformat's packet holds `size` bytes.
            let data = unsafe { std::slice::from_raw_parts(p.data, p.size as usize) };
            let pts = seconds(p.pts);
            let displays = packet_displays(codec, data, &mut vp9_colour)?;
            let payload = match container {
                // Read again when decoded. An edit list that shows media twice
                // would hand the decoder one packet twice: refused.
                Container::Mp4 => {
                    let offset = u64::try_from(p.pos).map_err(|_| {
                        MmError::invalid_input("MP4 packet without a file position")
                    })?;
                    if !offsets.insert(offset) {
                        return Err(MmError::unsupported(
                            "MP4 edit list that shows media more than once",
                        ));
                    }
                    Payload::Source {
                        offset,
                        len: data.len(),
                    }
                }
                Container::Matroska => {
                    held += data.len() as u64;
                    if held > max_input_bytes as u64 {
                        return Err(MmError::limit_exceeded(format!(
                            "WebM / Matroska packets exceed the {max_input_bytes} byte input limit"
                        )));
                    }
                    Payload::Owned(data.to_vec())
                }
            };
            let entry = Packet::new(
                payload,
                pts,
                displays,
                p.flags & ff::AV_PKT_FLAG_DISCARD != 0,
                p.flags & ff::AV_PKT_FLAG_KEY != 0,
            );
            if entry.shown && p.duration > 0 {
                let end = seconds(p.pts.saturating_add(p.duration));
                last_end = Some(last_end.map_or(end, |e: f64| e.max(end)));
            }
            packets.push(entry);
        }
    })();
    // SAFETY: allocated above, freed once.
    unsafe { ff::av_packet_free(&mut packet) };
    result?;
    // A read that failed can look like the end of the file to libavformat.
    if let Some(e) = demuxer.failure.take() {
        return Err(read_error(e, &format!("the {} file", container.name())));
    }

    // Times are player times: the clip starts at its first shown frame.
    let shown: Vec<f64> = packets.iter().filter(|p| p.shown).map(|p| p.pts).collect();
    let first = shown.iter().copied().fold(f64::INFINITY, f64::min);
    let first = if first.is_finite() { first } else { 0.0 };
    if first < 0.0 {
        return Err(MmError::invalid_input("video starts before time 0"));
    }
    let last = shown.iter().copied().fold(first, f64::max) - first;
    let duration = super::clip_duration(last_end.map(|end| end - first), last, shown.len());
    Ok(Track {
        codec,
        width,
        height,
        start_time: first,
        duration,
        config: match codec {
            VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Av1 => config,
            _ => None,
        },
        // `avc1` and `hvc1` keep every parameter set in the configuration
        // record; `avc3`, `hev1` and Matroska may also carry them in samples.
        inband_parameter_sets: matches!(codec, VideoCodec::H264 | VideoCodec::Hevc)
            && (container == Container::Matroska || matches!(&four_cc(tag), b"avc3" | b"hev1")),
        packets,
        vp9_colour,
        source,
    })
}

/// Refuse a WebM / Matroska file cut short. libavformat takes a cut for the
/// end of the file and returns the frames before it, but a cut file is not a
/// shorter video: a Segment of known size must fit in the file. (A Segment of
/// unknown size, as a live recorder writes, cannot be checked.)
fn matroska_segment_fits(source: &dyn ByteSource) -> Result<()> {
    let read = |at: u64, len: usize| -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        source
            .read_at(at, &mut buf)
            .map_err(|e| read_error(e, "the WebM / Matroska header"))?;
        Ok(buf)
    };
    // An EBML variable-size integer at `at`: its value (`None` when every
    // value bit is set, "unknown size") and its length.
    let vint = |at: u64| -> Result<(Option<u64>, u64)> {
        let first = read(at, 1)?[0];
        let len = first.leading_zeros() as usize + 1;
        if len > 8 {
            return Err(MmError::invalid_input("malformed WebM / Matroska header"));
        }
        let bytes = read(at, len)?;
        let mut value = u64::from(first) & (0xFF >> len);
        for &b in &bytes[1..] {
            value = (value << 8) | u64::from(b);
        }
        let unknown = value == (1u64 << (7 * len)) - 1;
        Ok(((!unknown).then_some(value), len as u64))
    };
    // EBML header (ID 1A 45 DF A3), then the Segment (ID 18 53 80 67).
    let (header, len) = vint(4)?;
    let header =
        header.ok_or_else(|| MmError::invalid_input("malformed WebM / Matroska header"))?;
    let segment = (4 + len)
        .checked_add(header)
        .ok_or_else(|| MmError::invalid_input("malformed WebM / Matroska header"))?;
    if read(segment, 4)? != [0x18, 0x53, 0x80, 0x67] {
        return Err(MmError::invalid_input(
            "WebM / Matroska file without a Segment",
        ));
    }
    let (size, len) = vint(segment + 4)?;
    if let Some(size) = size
        && (segment + 4 + len)
            .checked_add(size)
            .is_none_or(|end| end > source.len())
    {
        return Err(MmError::invalid_input(
            "truncated WebM / Matroska file: its Segment runs past the end",
        ));
    }
    Ok(())
}
