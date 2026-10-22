"""Geometry-only glTF Binary (.glb) writer/reader for the export bundle's mesh.

Positions, UVs and indices only. The single plain white material references no image and exists
because several importers drop TEXCOORD_0 from a primitive without a material. The container is
written directly so vertex order stays byte-identical to the input arrays.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

__all__ = ["write_glb", "read_glb"]

_GLB_MAGIC = 0x46546C67  # 'glTF'
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_COMPONENT_FLOAT = 5126
_COMPONENT_UINT = 5125
_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963
_MODE_TRIANGLES = 4


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * (-len(data) % 4)


def write_glb(
    path: str | Path,
    positions: np.ndarray,
    uv: np.ndarray,
    indices: np.ndarray,
    *,
    name: str = "mesh",
) -> int:
    """Writes (V,3) positions, (V,2) UVs and (F,3) indices as a single-primitive GLB; returns the file size."""
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    uv = np.ascontiguousarray(uv, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (V, 3), got {positions.shape}")
    if uv.shape != (len(positions), 2):
        raise ValueError(f"uv must be ({len(positions)}, 2), got {uv.shape}")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must be (F, 3), got {indices.shape}")

    buffer = bytearray()
    views = []
    for payload, target in (
        (indices.tobytes(), _TARGET_ELEMENT_ARRAY_BUFFER),
        (positions.tobytes(), _TARGET_ARRAY_BUFFER),
        (uv.tobytes(), _TARGET_ARRAY_BUFFER),
    ):
        views.append({"buffer": 0, "byteOffset": len(buffer), "byteLength": len(payload), "target": target})
        buffer += _pad4(payload)

    gltf = {
        "asset": {"version": "2.0", "generator": "3d_recon_sh_texture export"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 1, "TEXCOORD_0": 2},
                        "indices": 0,
                        "mode": _MODE_TRIANGLES,
                        "material": 0,
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _COMPONENT_UINT,
                "count": int(indices.size),
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "componentType": _COMPONENT_FLOAT,
                "count": int(len(positions)),
                "type": "VEC3",
                # Required by the spec for POSITION.
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {
                "bufferView": 2,
                "componentType": _COMPONENT_FLOAT,
                "count": int(len(uv)),
                "type": "VEC2",
            },
        ],
        # Textureless and white; see the module docstring.
        "materials": [
            {"name": "sh_texture", "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(buffer)}],
    }

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    bin_chunk = bytes(buffer)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    blob = (
        struct.pack("<III", _GLB_MAGIC, 2, total)
        + struct.pack("<II", len(json_chunk), _CHUNK_JSON)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), _CHUNK_BIN)
        + bin_chunk
    )
    Path(path).write_bytes(blob)
    return total


def read_glb(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of `write_glb`: (positions (V,3) f32, uv (V,2) f32, indices (F,3) u32). Reads
    only the layouts `write_glb` produces; not a general glTF importer."""
    raw = Path(path).read_bytes()
    magic, version, _total = struct.unpack_from("<III", raw, 0)
    if magic != _GLB_MAGIC or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary file")

    chunks: dict[int, bytes] = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunks[kind] = raw[offset + 8 : offset + 8 + length]
        offset += 8 + length
    gltf = json.loads(chunks[_CHUNK_JSON])
    blob = chunks[_CHUNK_BIN]

    dtypes = {_COMPONENT_FLOAT: np.float32, _COMPONENT_UINT: np.uint32}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}

    def fetch(accessor_index: int) -> np.ndarray:
        accessor = gltf["accessors"][accessor_index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        if "byteStride" in view or "sparse" in accessor:
            raise ValueError(f"{path}: strided or sparse accessors are not supported")
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        width = widths[accessor["type"]]
        array = np.frombuffer(
            blob, dtype=dtypes[accessor["componentType"]], count=accessor["count"] * width, offset=start
        )
        return array.reshape(-1, width) if width > 1 else array

    primitive = gltf["meshes"][0]["primitives"][0]
    positions = fetch(primitive["attributes"]["POSITION"])
    uv = fetch(primitive["attributes"]["TEXCOORD_0"])
    indices = fetch(primitive["indices"]).reshape(-1, 3)
    return positions, uv, indices
