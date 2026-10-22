"""Minimal KTX2 reader/writer for the export bundle's textures.

KTX2 is a container web loaders already parse (`ktx-parse`, three.js `KTX2Loader`,
...), and it carries the two things a bare `.npz` cannot: the pixel format the GPU
should allocate, and a supercompression scheme the loader undoes on the way to
`texSubImage`. The payload here is *not* a block-compressed (Basis/ASTC) texture,
because SH coefficients and a height field are numeric data, not colours, and an 8-bit
lossy codec on them shows up as banding in the parallax march.

Supercompression is **ZLIB (scheme 3), not Zstandard**, and that is a delivery choice
rather than a compression one: a browser decodes zlib natively with
`DecompressionStream("deflate")`, whereas zstd means shipping and instantiating a wasm
decoder before the first texture appears. Measured on the scan6 SH atlas, zlib -9 is
8.96 MB against zstd -12's 8.06, so the trade is 11% of the payload against a wasm
build dependency in every consumer. Note that three.js's own
`KTX2Loader` handles only schemes 0 and 2 and throws on this one - a consumer using it
needs the level data decoded before hand-off, or its own loader.

Only what the bundle needs is implemented: a single mip level, no cubemaps, no 3D
textures, float16 samples, optional zlib. That is a few hundred bytes of header rather
than a dependency on libktx, which ships no Linux Python wheel - and now no third-party
codec either, since zlib is in the standard library.

Formats are restricted to the ones a WebGL2 consumer can actually sample. three.js's
`KTX2Loader` FORMAT_MAP (checked against three.js dev) carries R/RG/RGBA at 8, 16F and
32F, but its only *3-channel* entries are the packed 32-bit ones (E5B9G9R9,
B10G11R11) and block-compressed formats - there is no plain R16G16B16_SFLOAT, so a
3-channel float texture would parse and then have no format to upload as. Pad to RGBA
instead; the padding is constant and costs ~7% once the compressor sees it.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

__all__ = ["write_ktx2", "read_ktx2", "KTX2_IDENTIFIER"]

KTX2_IDENTIFIER = b"\xabKTX 20\xbb\r\n\x1a\n"

# vkFormat by sample type and channel count. See the module docstring for why 3 is
# absent rather than merely unimplemented.
_VK_SFLOAT16 = {1: 76, 2: 83, 4: 97}  # R16_SFLOAT, R16G16_SFLOAT, R16G16B16A16_SFLOAT
_VK_UNORM8 = {1: 9, 2: 16, 4: 37}  # R8_UNORM, R8G8_UNORM, R8G8B8A8_UNORM
_VK_TO_LAYOUT = {**{v: (np.float16, k) for k, v in _VK_SFLOAT16.items()},
                 **{v: (np.uint8, k) for k, v in _VK_UNORM8.items()}}

_SUPERCOMPRESSION_ZLIB = 3  # an RFC 1950 zlib datastream, i.e. DecompressionStream("deflate")

# Khronos Data Format constants (khr_df.h), only the ones a linear float RGBA texture
# needs. Qualifiers 0xC0 = SIGNED | FLOAT.
_KHR_DF_MODEL_RGBSDA = 1
_KHR_DF_PRIMARIES_BT709 = 1
_KHR_DF_TRANSFER_LINEAR = 1
_KHR_DF_SAMPLE_FLOAT_SIGNED = 0xC0
# KHR_DF_CHANNEL_RGBSDA_*: alpha is 15, not 3.
_KHR_DF_CHANNELS = {1: (0,), 2: (0, 1), 4: (0, 1, 2, 15)}
_FLOAT_SAMPLE_LOWER = 0xBF800000  # -1.0f, the spec's convention for float channels
_FLOAT_SAMPLE_UPPER = 0x3F800000  # +1.0f

_WRITER = "3d_recon_sh_texture export"


def _basic_dfd(channels: int, unorm8: bool = False) -> bytes:
    """The Basic Data Format Descriptor block for `channels` x float16 (or x 8-bit UNORM),
    linear BT.709.

    The float16 block is byte-identical to what `ktx create --format R16G16B16A16_SFLOAT`
    emits, which is how it was checked - this block is what a loader reads to decide the
    texture's type, so getting it subtly wrong produces a file that validates as garbage
    rather than failing loudly. The UNORM8 block follows the same spec rules (unsigned,
    non-float qualifiers, sampleLower 0 / sampleUpper 255, transfer linear because the
    samples are data, not sRGB colour) but has not been compared against `ktx create`."""
    block_size = 24 + 16 * channels
    bits = 8 if unorm8 else 16
    words = [
        4 + block_size,  # dfdTotalSize
        0,  # vendorId (KHRONOS) | descriptorType (BASICFORMAT)
        (block_size << 16) | 2,  # descriptorBlockSize | versionNumber (1.3)
        (_KHR_DF_MODEL_RGBSDA) | (_KHR_DF_PRIMARIES_BT709 << 8) | (_KHR_DF_TRANSFER_LINEAR << 16),
        0,  # texelBlockDimension[0..3], stored as (size - 1); 1x1x1x1 -> all zero
        (bits // 8) * channels,  # bytesPlane[0..3]: one plane, `channels` samples per texel
        0,  # bytesPlane[4..7]
    ]
    for index, channel in enumerate(_KHR_DF_CHANNELS[channels]):
        qualifiers = 0 if unorm8 else _KHR_DF_SAMPLE_FLOAT_SIGNED
        words += [
            (channel | qualifiers) << 24 | ((bits - 1) << 16) | (index * bits),
            0,  # samplePosition[0..3]
            0 if unorm8 else _FLOAT_SAMPLE_LOWER,
            255 if unorm8 else _FLOAT_SAMPLE_UPPER,
        ]
    return struct.pack(f"<{len(words)}I", *words)


def _key_value_data(entries: dict[str, str]) -> bytes:
    """KVD entries, each a u32 length + NUL-terminated key + value, padded to 4 bytes.
    The spec requires the keys be sorted by codepoint, so they are."""
    out = bytearray()
    for key in sorted(entries):
        payload = key.encode() + b"\x00" + entries[key].encode() + b"\x00"
        out += struct.pack("<I", len(payload)) + payload
        out += b"\x00" * (-len(payload) % 4)
    return bytes(out)


def write_ktx2(path: str | Path, data: np.ndarray, *, level: int = 9) -> int:
    """Writes `data` as a single-level KTX2 texture and returns the file size.

    `data` is (H, W, C) for a 2D texture or (L, H, W, C) for a 2D array texture. uint8
    input is written as UNORM8 (R8/RG8/RGBA8_UNORM); anything else is cast to float16.
    Row 0 is the top row (KTXorientation "rd", KTX2's default), i.e. the atlas
    convention - a GL consumer flips it exactly as it flips a PNG.

    `level` is the zlib compression level, 1-9; 0 stores the level uncompressed.
    """
    unorm8 = np.asarray(data).dtype == np.uint8
    formats = _VK_UNORM8 if unorm8 else _VK_SFLOAT16
    data = np.ascontiguousarray(np.asarray(data, dtype=np.uint8 if unorm8 else np.float16))
    # layerCount 0 is KTX2 for "not an array texture"; a 4-D input asks for an array,
    # even a one-layer one, so a degree-0 fit still loads as sampler2DArray.
    is_array = data.ndim == 4
    if data.ndim == 3:
        data = data[None]
    if data.ndim != 4:
        raise ValueError(f"expected (H, W, C) or (L, H, W, C), got shape {data.shape}")
    layers, height, width, channels = data.shape
    if channels not in formats:
        raise ValueError(f"{channels}-channel {data.dtype} has no WebGL-safe vkFormat; pad to 1, 2 or 4")

    payload = data.tobytes()
    if level > 0:
        level_bytes = zlib.compress(payload, level)
        scheme = _SUPERCOMPRESSION_ZLIB
    else:
        level_bytes, scheme = payload, 0

    dfd = _basic_dfd(channels, unorm8)
    kvd = _key_value_data({"KTXorientation": "rd", "KTXwriter": _WRITER})

    # Header, then the one-entry level index, then DFD, KVD, then the level itself.
    # Uncompressed levels need mip padding to lcm(texel size, 4); a supercompressed
    # level has an alignment of 1, and the only level here starts at a 4-byte boundary
    # anyway because both the DFD and the KVD are 4-byte multiples.
    dfd_offset = 80 + 24  # 12-byte identifier + 68-byte header + a one-entry level index
    kvd_offset = dfd_offset + len(dfd)
    level_offset = kvd_offset + len(kvd)

    header = KTX2_IDENTIFIER + struct.pack(
        "<13I2Q",
        formats[channels],
        1 if unorm8 else 2,  # typeSize: bytes per channel
        width,
        height,
        0,  # pixelDepth: 0 for a 2D texture
        layers if is_array else 0,  # layerCount: 0 means "not an array texture"
        1,  # faceCount
        1,  # levelCount
        scheme,
        dfd_offset,
        len(dfd),
        kvd_offset,
        len(kvd),
        0,  # supercompressionGlobalData offset (zlib needs none)
        0,  # ... and length
    )
    level_index = struct.pack("<3Q", level_offset, len(level_bytes), len(payload))

    path = Path(path)
    path.write_bytes(header + level_index + dfd + kvd + level_bytes)
    return len(header) + len(level_index) + len(dfd) + len(kvd) + len(level_bytes)


def read_ktx2(path: str | Path) -> np.ndarray:
    """Inverse of `write_ktx2`: returns (H, W, C), or (L, H, W, C) for arrays.

    Float16 files come back as float32, because every consumer here does arithmetic on
    the result and the file is what stores 16 bits. UNORM8 files come back as the raw
    uint8 samples: what they mean (e.g. a bundle's per-coefficient offset/scale) lives
    outside the texture."""
    raw = Path(path).read_bytes()
    if raw[:12] != KTX2_IDENTIFIER:
        raise ValueError(f"{path} is not a KTX2 file")
    (vk_format, _type_size, width, height, depth, layer_count, face_count, level_count,
     scheme, _dfd_off, _dfd_len, _kvd_off, _kvd_len, _sgd_off, _sgd_len) = struct.unpack_from(
        "<13I2Q", raw, 12
    )
    if vk_format not in _VK_TO_LAYOUT:
        raise ValueError(f"{path}: vkFormat {vk_format} is not one this reader writes")
    if depth or face_count != 1 or level_count > 1:
        raise ValueError(f"{path}: only single-level 2D (array) textures are supported")

    offset, byte_length, uncompressed_length = struct.unpack_from("<3Q", raw, 80)
    level = raw[offset : offset + byte_length]
    if scheme == _SUPERCOMPRESSION_ZLIB:
        level = zlib.decompress(level, bufsize=uncompressed_length)
    elif scheme != 0:
        raise ValueError(f"{path}: supercompression scheme {scheme} is not supported")

    dtype, channels = _VK_TO_LAYOUT[vk_format]
    array = np.frombuffer(level, dtype=dtype)
    array = array.astype(np.float32) if dtype == np.float16 else array.copy()
    if layer_count == 0:
        return array.reshape(height, width, channels)
    return array.reshape(layer_count, height, width, channels)
