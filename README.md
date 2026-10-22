# snap3d

A pipeline that turns photos of an object into a mesh + SH texture that renders directly on the web (WebGL).
It reconstructs a mesh with Gaussian-splatting-based surface reconstruction (Fast-PGSR), simplifies it to about 50k faces and unwraps its UVs.
For every texel it fits spherical-harmonics (SH) coefficients, so view-dependent color (highlights, reflections) is stored in the texture.
The result is a low-poly mesh plus two textures, so it can be rendered with shaders alone, without a neural network.

## Requirements

- **An NVIDIA GPU with a recent driver** (`nvidia-smi` must work). The surface reconstruction step uses CUDA kernels.
- **Linux** (tested on Ubuntu). Windows is untested; WSL2 with an NVIDIA GPU may work. macOS is not supported (no CUDA).
- `conda` and `git`.
- A Hugging Face account with access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3) checkpoint.

## Run

```bash
git clone --recursive https://github.com/teo646/snap3d.git
cd snap3d
./setup.sh                 # creates the conda envs snap3d and fastgs + builds the CUDA extensions
conda activate snap3d
hf auth login              # SAM3 access is required (huggingface.co/facebook/sam3)
```

### Quick start

`data/example/statue` contains 55 photos of a statue.

```bash
python -m pipeline.orchestrator.runner \
    --config pipeline/orchestrator/configs/photo_pipeline.yaml \
    --input data/example/statue \
    --text statue \
    --run-id example_statue
```

For your own photos, pass the photo folder as `--input` and an English word describing the object as `--text`.
Results are written to `data/runs/<run_id>/`, one folder per stage.

### Viewing the result

The final bundles are `data/runs/<run_id>/<NN>_export/<run_id>_unorm8.snap3d` (small) and `..._float16.snap3d` (higher precision).
`view_bundle.py` renders using only the files in the bundle.

```bash
pip install -e ".[viewer]"   # already installed if you used setup.sh
python viewers/view_bundle.py data/runs/example_statue/*_export/example_statue_unorm8.snap3d
```

- Controls: left-drag to rotate, wheel to zoom, WASD/arrow keys to move, Q/E to tilt the up axis, R to reset the view, ESC to quit
- The path can be the `.snap3d` folder or its parent `export` folder.
- With `--modify_initial_cam`, the camera at exit is saved as the initial view in the bundle's `config.json`.

## Pipeline

```
photos - SfM - masking - surface recon - hole closing - simplification - UV unwrap - baking - SLF fit - export
```

| Stage | What it does | Technology |
|---|---|---|
| ingest | Camera poses and sparse point cloud | COLMAP |
| resize_dataset | Converts to pinhole images at training resolution | |
| masking | Per-photo object mask (text prompt) | SAM3 |
| surface_recon | High-resolution mesh | Fast-PGSR (Gaussian splatting -> TSDF fusion) |
| hole_close, decimation | Closes holes and simplifies to about 50k faces | PyMeshLab |
| uv_unwrap | UV atlas | xatlas |
| baking | High-res -> low-res height map | Ray casting |
| slf | Per-texel SH coefficient fit | Conjugate gradient + normal equations |
| export | Render bundle `mesh.glb`, `sh.ktx2`, `height.ktx2` | glTF / KTX2 |
