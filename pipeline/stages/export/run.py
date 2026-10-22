"""Export stage wrapper: uv_mesh + slf -> render-ready `<run_id>_<storage>.snap3d/` folders.

Terminal stage. Reads nothing the earlier stages did not already write; its whole job
is to put the render-time inputs, and the handful of constants a renderer cannot infer
from them, in one folder with no reference back to this repo's layout.

Two bundles are written, one per SH storage (`unorm8` and `float16`; there is no param to
choose one). Each goes in a `.snap3d` subfolder rather than in the stage directory
itself, so that a deliverable stays one nameable thing next to the stage's own
`manifest.json` instead of being mixed in with it. The folders are named after the run -
the stage reads that off its own output directory's parent, which is the run dir the
orchestrator made - so a bundle carried out of this tree still says which capture it is. See
`pipeline/core/render_bundle.py` for the layout.

`num_steps` is not a param here: it has to equal the value the SH coefficients were fit
against, so it is read off the slf stage's own manifest rather than restated. The
`posed_images` input is used for one thing only - aiming the bundle's initial view at
the side the object was photographed from - and is optional; without it the view falls
back to the config's fixed azimuth/elevation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.core.render_bundle import (
    BUNDLE_FILES,
    SH_STORAGES,
    RenderBundleConfig,
    bundle_dir_name,
    write_render_bundle,
)
from pipeline.core.slf_fit import SLFFitConfig
from pipeline.schemas.manifest import StageManifest, read_manifest
from pipeline.schemas.posed_images import PosedImageSet
from pipeline.schemas.texture_maps import HeightMap, SHTextureMap
from pipeline.schemas.uv_mesh import UVMesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    if "sh_storage" in params:
        raise ValueError(
            "export no longer takes `sh_storage`: it writes a bundle for every storage in "
            f"{SH_STORAGES} - drop the param"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    uv_dir = input_dirs["uv_mesh"]
    slf_dir = input_dirs["slf"]

    uv_mesh = UVMesh.load(uv_dir / "mesh_uv.obj", texture_resolution=_uv_texture_resolution(uv_dir))
    sh_map = SHTextureMap.load(slf_dir / "sh_texture_map.npz")
    height_map = HeightMap.load(slf_dir / "height_map.png")

    config = RenderBundleConfig(num_steps=_slf_num_steps(slf_dir), **params)
    camera_centers = _camera_centers(input_dirs)
    # `<run>/NN_export` -> the run's name. A stage run straight from the CLI into some
    # other directory just gets that directory's parent, which is the best guess there is.
    run_name = output_dir.resolve().parent.name

    # Paths relative to the stage dir, i.e. inside the bundle folder - the manifest is
    # the stage's, and it sits beside the bundles rather than in them.
    outputs: dict[str, str] = {}
    for sh_storage in SH_STORAGES:
        dir_name = bundle_dir_name(run_name, sh_storage)
        bundle_dir = output_dir / dir_name
        bundle = write_render_bundle(
            uv_mesh, sh_map, height_map, bundle_dir, sh_storage, config, camera_centers=camera_centers
        )
        sizes = {name: (bundle_dir / filename).stat().st_size for name, filename in BUNDLE_FILES.items()}
        print(
            f"  {dir_name}: {bundle['mesh']['faces']} faces | SH degree {bundle['sh']['degree']} | "
            f"{bundle['texture_resolution'][0]}x{bundle['texture_resolution'][1]} | "
            f"num_steps {bundle['height']['num_steps']} | "
            f"zeroed {bundle['sh']['zeroed_fraction'] * 100:.1f}% of higher-order coeffs, "
            + (f"{bundle['sh']['mantissa_bits']}-bit mantissa | " if sh_storage == "float16" else "unorm8 | ")
            + f"{sum(sizes.values()) / 1e6:.1f} MB (sh {sizes['sh'] / 1e6:.1f}, "
            f"height {sizes['height'] / 1e6:.1f}, mesh {sizes['mesh'] / 1e6:.1f})"
        )
        outputs.update({f"{name}_{sh_storage}": f"{dir_name}/{filename}" for name, filename in BUNDLE_FILES.items()})
    return finalize_stage("export", output_dir, params, input_dirs, outputs=outputs, start_time=t0)


def _camera_centers(input_dirs: dict[str, Path]) -> np.ndarray | None:
    ingest_dir = input_dirs.get("posed_images")
    if ingest_dir is None:
        return None
    return np.stack([img.pose.camera_center() for img in PosedImageSet.load(ingest_dir).images])


def _uv_texture_resolution(uv_dir: Path) -> tuple[int, int]:
    manifest = read_manifest(uv_dir)
    if manifest is not None and "texture_resolution" in manifest.params:
        w, h = manifest.params["texture_resolution"]
        return (w, h)
    return (1024, 1024)


def _slf_num_steps(slf_dir: Path) -> int:
    """The march depth the coefficients were fit with. A config that leaves `num_steps`
    out of the slf params fit with SLFFitConfig's default, so that is the fallback -
    not an error, and not this stage's own default."""
    manifest = read_manifest(slf_dir)
    if manifest is not None and "num_steps" in manifest.params:
        return int(manifest.params["num_steps"])
    # slf_bound republishes slf's map and carries slf's num_steps in `extra`.
    if manifest is not None and "num_steps" in manifest.extra:
        return int(manifest.extra["num_steps"])
    return SLFFitConfig().num_steps


def main() -> None:
    stage_cli("export", run)


if __name__ == "__main__":
    main()
