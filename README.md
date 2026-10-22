# 3d_recon_sh_texture

여러 장의 사진으로부터 3D 표면과 SH(구면조화) 기반 텍스처를 만들어, 웹(WebGL)에서 렌더링 가능한 **mesh** + **height map** + **SH texture map**을 생성하는 파이프라인.

```
사진 여러 장 (다시점, 단일 물체, 고정 조명)
  → [Ingest/SfM]        카메라 포즈 + sparse point cloud   (COLMAP)
  → [Resize Dataset]    학습 해상도 PINHOLE 이미지+카메라   (다운스케일 + 카메라 스케일, dense stereo 없음)
  → [Masking]           per-view 전경 마스크               (SAM3 text prompt)
  → [Surface Recon]     high-poly mesh                    (Fast-PGSR: Gaussian Splatting → TSDF fusion)
  → [Hole Close]        작은 구멍을 막은 high-poly mesh     (PyMeshLab meshing_close_holes)
  → [Decimation]        low-poly mesh                     (PyMeshLab quadric edge collapse)
  → [UV Unwrap]         uv_mesh                           (xatlas)
  → [Baking]            baseline height_map               (high-poly → low-poly UV ray-cast)
  → [SLF fitting]       height_map + sh_texture_map       (CG + AtA/Atb 텍셀별 SH fit, POM 시차 보정)
```

## 레포 구조

```
pipeline/
  core/           # 순수 알고리즘 (디렉토리 IO 없음)
    ingest/
      photo_sfm.py         # 사진 -> 카메라 포즈 + sparse cloud (COLMAP 서브프로세스)
    resize_dataset.py      # posed_images -> 학습 해상도 PINHOLE 이미지 + 카메라 (dense stereo 없음)
    masking_sam3.py        # SAM3 text-prompted masking (파일명 순서로 한 비디오, 놓친 프레임은 같은 텍스트로 재시도)
    colmap_dataset.py      # dense/ -> Fast-PGSR/FastGS 데이터셋 레이아웃 빌더
    surface_recon_fastpgsr.py # Fast-PGSR(third_party/FastGS) 래핑
    hole_close.py          # decimation 전에 작은 구멍을 삼각분할로 막음
    decimation.py          # quadric edge collapse
    uv_unwrap.py           # xatlas 언랩 (gutter를 pack 단계에서 예약)
    baking.py              # high-poly -> low-poly UV ray-cast, baseline height map
    uv_geometry.py         # UV atlas 래스터화 + 정점별 frame 속성 (half-texel 컨벤션)
    camera_raster.py       # low-poly mesh를 headless OpenGL로 래스터화 + POM resolve
    slf_shaders.py         # 학습/렌더가 공유하는 GLSL (frame, POM march, SH eval)
    sh_math.py             # real SH basis + HSH remap + CG bilinear-texel 복원 (torch)
    slf_fit.py             # 텍셀별 SH 계수 피팅 (CG + AtA/Atb, POM 시차 보정)
    orbit_camera.py        # 캡처의 up 축을 중심으로 도는 orbit 카메라 + GL 행렬
    render_bundle.py       # 렌더에 필요한 것만 한 폴더로 - 다른 프로젝트가 참조할 산출물
    ktx2.py                # KTX2 컨테이너 write/read (RGBA16F/RG16F, RGBA8 UNORM + zlib)
    gltf.py                # 지오메트리 전용 .glb write/read

  schemas/        # 단계 간 데이터 포맷 (pydantic / dataclass)
    manifest.py, posed_images.py, point_cloud.py, mesh.py, uv_mesh.py, texture_maps.py

  stages/         # core를 감싸는 디렉토리 입출력 어댑터 (core와 1:1)
    <stage>/run.py    # run(input_dirs, output_dir, params) -> StageManifest
                      # 단독 실행: python -m pipeline.stages.<stage>.run \
                      #     --input name=path ... --output DIR --params '{...}'

  orchestrator/
    runner.py                    # yaml의 stages 리스트만 보고 순서/의존성 결정
    configs/photo_pipeline.yaml  # 사진 파이프라인 (surface_recon = Fast-PGSR)

viewers/          # 각 단계 결과를 눈으로 확인하는 대화형 뷰어 (moderngl+pygame / open3d)
  view_bundle.py  # export 번들만 보고 렌더 - 소비 측 관점
  view_slf.py, view_uv_unwrap.py, view_mesh.py
  _common.py      # run 디렉토리 탐색 + 캡처 up 축 (카메라는 core/orbit_camera.py)

third_party/FastGS/ # fastgs/FastGS (fast-pgsr 브랜치) submodule - 사진 파이프라인 surface_recon

data/runs/<run_id>/
  01_ingest/            images/, cameras.json, sparse_points.ply
  02_resize_dataset/   (산출물은 _workspace/dense/{images,sparse})
  03_masking/           masks/
  04_surface_recon/     mesh_highpoly.obj
  05_hole_close/        mesh_highpoly.obj  (구멍을 막은 것, 04와 같은 파일명)
  06_decimation/        mesh_lowpoly.obj
  07_uv_unwrap/         mesh_uv.obj
  08_baking/            height_map_baseline.png (+.npy)
  09_slf/               height_map.png (+.npy), sh_texture_map.png (+.npz)
  10_export/<run_id>_unorm8.snap3d/, <run_id>_float16.snap3d/  렌더용 번들 mesh.glb + sh.ktx2 + height.ktx2 + config.json (아래 참조)
  (각 디렉토리에 manifest.json)

```

### 설계 원칙

- **ingest와 나머지 단계는 분리**되어 있습니다. `posed_images` 스키마(`images/` + `cameras.json`)만 만족하면 뒤 단계는 입력이 무엇이었는지 모릅니다.
- **masking은 resize_dataset *다음에*** 실행됩니다. SAM3가 Fast-PGSR이 실제로 쓰는 (필요시 `max_image_size`로 다운스케일된) 이미지를 그대로 세그먼트하므로 마스크 해상도를 어디서도 맞춰줄 필요가 없습니다. masking 스테이지는 하나뿐입니다: 입력 폴더의 사진을 파일명 순서로 마스킹해 `masks/<이미지 이름>.png`를 씁니다 (COLMAP `mask_path` 형식). 그래서 SfM 전 마스킹과 SfM 후 마스킹이 같은 스테이지입니다.
- 마스크는 COLMAP에는 쓰이지 않고, `surface_recon`이 팽창시켜 alpha 채널로 합성한 뒤 **render/TSDF-fusion 단계에서만** 사용합니다. 마스크 없이 fusion하면 배경 전체가 메시에 섞여 수십 GB까지 치솟다 OOM으로 죽습니다.
- 모든 stage 디렉토리에 `manifest.json`(params + 입력 해시 + timestamp)이 남고, runner는 이를 이용해 입력이 바뀌지 않은 stage를 스킵합니다 (`--force`로 무시).
- 파이프라인 순서/연결은 코드가 아니라 `configs/*.yaml`의 `stages` 리스트가 결정합니다.

## 환경 설정

Python 3.12 + GPU 지원 COLMAP(conda-forge)을 쓰는 `sh-texture` conda 환경. GPU는 RTX 50 시리즈(Blackwell, sm_120) 기준으로 검증했습니다.

```bash
conda env create -f environment.yml
conda activate sh-texture

pip install -e ".[dev]"        # numpy, opencv, pydantic, open3d, pymeshlab, xatlas, trimesh, pytest
pip install -e ".[masking]"    # + torch/torchvision/SAM3 (무거움, GPU 필요)
pip install -e ".[viewer]"     # + pygame (viewers/ 실행용)
```

> `environment.yml`이 `colmap`뿐 아니라 `libfaiss`/`openimageio`를 특정 build로 pin하는 이유: conda-forge의 `colmap`이 실제로 링크하는 라이브러리를 런타임 의존성으로 선언하지 않아, `colmap`만 설치하면 `error while loading shared libraries`로 깨집니다. 재발하면 `ldd $(which colmap) | grep "not found"`로 확인하세요.

SAM3 체크포인트는 Hugging Face에서 자동 다운로드됩니다 (`facebook/sam3`, access-gated - 접근 권한 요청 후 `huggingface-cli login` 한 번). 로컬 체크포인트는 `masking.params.checkpoint_path`로 지정합니다.

서브모듈까지 받으려면: `git submodule update --init --recursive`

일부 스테이지는 별도 conda 환경의 python을 서브프로세스로 호출합니다(`fastgs`, `da3`, `shiq`, `mpface`). 기본값은 현재 환경과 같은 `envs/` 아래의 같은 이름 환경이고, `SNAP3D_FASTGS_PYTHON`, `SNAP3D_DA3_PYTHON`, `SNAP3D_SHIQ_PYTHON`, `SNAP3D_MPFACE_PYTHON` 환경변수로 경로를 바꿀 수 있습니다. highlight_removal 스테이지는 `third_party/SHIQ`(https://github.com/KosRud/SHIQ)를 직접 clone해 두어야 합니다.

## 실행

```bash
python -m pipeline.orchestrator.runner \
    --config pipeline/orchestrator/configs/photo_pipeline.yaml \
    --input /path/to/photo/dir \
    --text "object description for SAM3"
```

- 결과는 `data/runs/photo_<timestamp>/`에 stage별로 쌓입니다.
- `--text "설명"`: 자체 `text_prompt`가 없는 masking 단계들의 SAM3 프롬프트. **생략하면 masking 단계를 건너뜁니다** (배경을 걷어낼 필요가 없는 촬영용).
- `--until <stage_id>`: 해당 stage까지만 실행. `--run-id NAME`: run 디렉토리 이름 고정. `--force`: 캐시 무시.

결과 확인:

```bash
python viewers/view_slf.py data/runs/<run_id>        # 최종 SH 텍스처
python viewers/view_uv_unwrap.py data/runs/<run_id>  # UV 언랩 품질
```

## `third_party/patches`: 서브모듈 패치

`third_party/` 아래 서브모듈(Metric3D, FastGS)은 전부 upstream 커밋에
고정돼 있고, 필요한 수정은 **작업 트리에 직접** 가해집니다. 상위 저장소는 서브모듈의
커밋 SHA만 기록하므로 그 수정들은 버전 관리 밖에 있습니다 — `third_party/patches/*.patch`가
그것들을 담은 실체이고, 새로 클론한 뒤 이 스크립트를 돌려야 동작하는 체크아웃이 됩니다.

```bash
git submodule update --init --recursive
./third_party/patches/apply.sh            # 적용
./third_party/patches/apply.sh --check    # 적용 여부 확인
./third_party/patches/apply.sh --revert   # 되돌리기
```

각 패치는 생성 기준 커밋이 스크립트에 박혀 있고 적용 전에 검사합니다. 서브모듈이 다른
리비전에 있으면 조용히 엉뚱한 줄에 붙는 대신 건너뛰고 알려줍니다. 패치를 수정한 뒤에는
다시 뽑아 커밋하세요:

```bash
git -C third_party/<name> diff -- . ':!*__pycache__*' ':!*.egg-info*' ':!build/' \
    > third_party/patches/<name>.patch
```

패치 내용과 그 이유는 각 수정 위치의 주석, 그리고 아래 절들에 적혀 있습니다.

## `third_party/FastGS`: 학습을 가속한 PGSR (사진 파이프라인 기본값)

[fastgs/FastGS](https://github.com/fastgs/FastGS)의 `fast-pgsr` 브랜치는 PGSR에 FastGS의
densification 가속을 얹은 것입니다. loss map 임계값과 split/clone을 각각 담당하는 두 개의
gradient 임계값으로 어떤 Gaussian을 남길지 결정하고, SH의 저차/고차 계수를 서로 다른 학습률로
훈련합니다. 산출물은 여전히 PGSR의 것입니다 — planar Gaussian을 TSDF로 융합한
`mesh/tsdf_fusion_post.ply`, 최대 연결 요소만.

그 계약이 같아서 wrapper가 얇습니다. Fast-PGSR은 PGSR의 인자를 전부 상위집합으로 받고 같은
경로에 같은 이름으로 메시를 쓰며 `post_process_mesh(cluster_to_keep=1)`도 동일하므로,
데이터셋 빌더와 마스크 render 데이터셋은 `pipeline/core/colmap_dataset.py`에 있습니다.

**별도 `fastgs` conda env가 필수입니다.** 두 레포가 `diff-plane-rasterization`, `fused-ssim`,
`simple-knn`이라는 **같은 이름에 서로 다른 소스**를 담은 패키지를 배포해서, 한쪽을 다른 쪽의
env에 설치하면 rasterizer가 조용히 뒤바뀝니다. `pgsr` env를 클론해 만들면 검증된
torch 2.11+cu128 빌드를 그대로 물려받습니다.

```bash
conda create -n fastgs --clone pgsr -y && conda activate fastgs
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
for m in diff-gaussian-rasterization_fastgs diff-plane-rasterization fused-ssim simple-knn; do
  pip install third_party/FastGS/submodules/$m --no-build-isolation --force-reinstall --no-deps
done
```

분리가 됐는지는 두 env의 `diff_plane_rasterization` `.so` 해시가 다른지로 확인합니다.

상류 대비 동작이 달라지는 패치가 하나 있습니다. `arguments/__init__.py`가
`multi_view_weight_until_iter=10000`을 선언하지만 상류 `train.py`는 이 값을 읽지 않아서,
multi-view loss가 7000 iter부터 마지막까지 켜져 있습니다. 이 조건을 연결하면 boots에서
30000 iter 학습이 50:00 → 20:23으로 줄고 메시는 동등했습니다 (정점 544만 vs 563만, 표면적 차 3%).
메시는 상류 `run_tnt.py`가 부르는 `scripts/render_tnt.py`로 뽑되, PGSR과 같은 마스크 depth
제거 두 줄을 추가했습니다 (최상위 `render.py`는 `mesh_path` 미정의로 끝까지 돌지 않습니다).

## `slf_fit.py`: SH 텍스처 피팅

teo646/surface-light-fields-for-planar-paintings의 `53f5787`(POM 기반으로 안정화된, height-map 최적화 이전 버전)을 이식했습니다. 원본은 평면 그림 한 장이라 전역 프레임 하나로 충분했지만 여기서는 임의의 mesh이므로:

- **텍셀별 tangent frame**: UV 파라미터화에서 유도한 프레임을 POM march와 SH 방향 인코딩에 모두 사용합니다. 이 프레임은 **텍스처로 굽지 않고** 정점 속성(normal, 정규화하지 않은 dP/du, dP/dv)으로 실어 프래그먼트 셰이더에서 재구성합니다 (`uv_geometry.vertex_frame_attributes` ↔ `slf_shaders.uv_frame`). 아틀라스가 바리센트릭 보간하는 것과 같은 배열이라 값이 정확히 일치하고, 텍스처 4장(tangent/bitangent/normal/xy_per_height)이 페이로드에서 빠집니다.
- **GPU forward pass** (`camera_raster.py`): 원본의 moderngl "UV pass"를 일반화 - low-poly mesh를 headless OpenGL로 z-buffer 래스터화해 닫힌 물체의 self-occlusion을 처리합니다. POM march GLSL은 `slf_shaders.RESOLVE_GLSL` 하나를 학습과 뷰어가 그대로 공유합니다.
- **CG + AtA/Atb** (`sh_math.py`): `bilinear_neighbors` / `cg_solve_bilinear_texels` / 실구면조화 basis를 torch로 이식하고, 원본의 HSH(반구→전구) remap도 그대로 씁니다. `view_local`이 이미 텍셀 normal 기준이라 remap의 정의역이 성립하고, 같은 degree로 반구 방향 디테일을 더 조밀하게 표현할 수 있어 기본 `sh_degree=1`이 성립합니다.

- **depth test** (`depth_test_tolerance`): POM은 픽셀을 같은 표면 위에서 옮길 뿐 *다른 표면*으로 옮기지 못합니다. decimation이 실루엣을 움직이므로, 객체의 두 부분이 겹쳐 보이는 곳에서는 앞 부분의 테두리 픽셀이 뒤 부분 삼각형에 래스터화됩니다. 같은 패스에서 resolved uv가 가리키는 표면(lowpoly 점을 march가 도달한 height만큼 법선 방향으로 민 점)의 카메라 depth를 함께 출력하고, highpoly 메시(`hole_close`, baking이 구운 원본)를 같은 카메라로 렌더한 depth와 비교해 상대 차이가 허용치를 넘는 픽셀을 버립니다. highpoly가 없는 픽셀(빈 공간으로 삐져나온 실루엣)도 버려집니다. 0이면 꺼지고 `highpoly_mesh` 입력도 필요 없습니다. 비교 기준은 `depth_reference`로 고릅니다: `highpoly`는 위의 메시 depth, `gaussians`는 slf 시작 시 학습된 Fast-PGSR 가우시안을 각 카메라로 렌더한 `plane_depth`(train.py가 loss에, render_tnt.py가 TSDF에 쓰는 그 depth)입니다. 메시는 그 depth들을 2mm TSDF 복셀로 평균낸 것이라 보풀 섬유나 얇은 테두리 가장자리가 지워지고, 사진에서 그것이 안감 앞을 가리는 픽셀은 메시 기준으로는 "안감"으로 통과합니다. 가우시안 depth는 `fastgs` env 서브프로세스(`core/_fastgs_depth_worker.py`)로 `09_slf/_gaussian_depth/`에 렌더되고 피팅 후 삭제됩니다 (12MP float32 × 뷰 수라 남겨두면 하위 스테이지가 매번 해싱합니다).

주요 파라미터: `sh_degree`(0-3), `num_steps`(POM march 층 수), `ridge`, `min_observations_per_texel`, `min_precision_ridge_ratio`, `depth_test_tolerance`, `depth_reference`, `device`.

## `export`: 렌더용 번들

마지막 stage는 계산을 하지 않습니다. 렌더에 필요한 것만, 이 레포의 디렉토리 구조를 몰라도
읽을 수 있는 형태로 `.snap3d` 폴더에 모읍니다. SH 저장 방식이 다른 두 벌(`unorm8`, `float16`)을 항상 함께 만들고, 저장 방식을 고르는 옵션은 없습니다. 웹 렌더러가 이미 로더를 가진 포맷(.glb /
.ktx2)으로 네 개, 그리고 이 네 개에서 유도할 수 있는 것은 아무것도 들어가지 않습니다
(프리뷰 PNG도, OBJ 사본도, 정점 프레임도).

```
10_export/
  <run_id>_<storage>.snap3d/   예: photo_20260904_000809_unorm8.snap3d, photo_20260904_000809_float16.snap3d
    mesh.glb      POSITION (V,3) f32, TEXCOORD_0 (V,2) f32, indices (F,3) u32
    sh.ktx2       배열 텍스처, 계수 k당 한 레이어. rgb = 계수의 RGB, a = 0 (패딩).
                  config.json `sh.storage`가 "unorm8"이면 RGBA8 UNORM이고
                  계수 = sh.offset[k][c] + sh.scale[k][c] * sample (sample은 GPU가 주는 [0,1] 값),
                  "float16"이면 RGBA16F에 계수 그대로
    height.ktx2   RG16F. r = 변위(height.units), g = 아틀라스 커버리지 (1/0)
    config.json   up_vector, sh{...}, texture_resolution, height{...}, initial_camera
  manifest.json   stage 자신의 것. 번들 밖에 있습니다 (번들에는 이 레포 흔적이 없어야 하므로)
```

- **`.snap3d`는 폴더 이름 규약이지 컨테이너가 아닙니다.** 폴더 이름은 run 이름을 그대로 씁니다
  (`photo_20260904_000809.snap3d`). 이 트리 밖으로 복사해 나가도 어느 캡처인지가 남게 하려는
  것으로, run id는 번들 자체가 달리 들고 있지 않은 유일한 출처 정보입니다. 그리고 아카이브가
  아니라 평범한 파일 네 개이고 각각 따로 fetch됩니다 - 그게 목적입니다. 디렉토리 URL 하나를 받은 소비자는 요청
  세 개를 병렬로 던지고, 각 페이로드는 아카이브의 한 조각이 아니라 그 자체로 캐시되고
  재검증됩니다 (아카이브라면 어느 하나를 읽기 전에 통째로 도착해야 합니다). 확장자는 "이
  폴더 하나가 완결된 결과물"이라는 표시일 뿐이며(`.app`, `.framework`와 같은 방식), 풀
  아카이브도 파싱할 오프셋도 없고 진입점은 여전히 `config.json`입니다. stage 출력 디렉토리
  자체가 아니라 그 안의 하위 폴더인 이유는, stage의 `manifest.json`과 섞이지 않게 하기
  위해서입니다.

- **두 파일 모두 이 레포가 아니라 각 포맷 자신의 규약을 따릅니다.** glTF의 UV 원점은 이미지
  좌상단이고 KTX2는 row 0을 먼저 저장하므로, `TEXCOORD_0`은 `UVMesh.uv` 기준으로 v가 뒤집힌
  채로, 텍스처는 위쪽 행부터 기록됩니다. 즉 **glTF/KTX2만 그대로 따르는 소비자는 아무 예외
  처리 없이 맞습니다.** 대신 이 레포의 v-up 셰이더를 재사용하는 `view_bundle.py`가 로드
  시점에 두 번 되돌립니다 (`v <- 1 - v`, 그리고 `to_gl_texture`의 행 뒤집기). 직접 셰이더를
  쓰는 소비자는 되돌리는 대신 `xy_per_height`의 `-H` 항 부호를 바꾸면 됩니다.
- **정점 프레임은 싣지 않고 다시 계산합니다.** 셰이더는 normal과 정규화하지 않은 raw dP/du,
  dP/dv도 받는데, 셋 다 position+uv+indices만의 함수입니다 (`uv_geometry.vertex_normals_for`
  + `compute_vertex_tangents`, 정점 65k에 28 ms). 실으면 지오메트리 페이로드가 3배가 되고,
  아끼는 건 체감되지 않는 로드 시간뿐입니다. 대신 **소비 측이 이 두 함수를 정확히 재현해야
  합니다**: **면적 가중이 아니라 각도 가중** face normal (trimesh의 smooth vertex normal이
  인접 면 normal을 그 면의 정점 내각으로 가중합니다. 이 레포 메시로 실측: 각도 가중은 출력을
  1.3e-14까지 재현, 면적 가중은 최대 1.6, 균등 평균은 0.83 어긋납니다), Lengyel tangent를
  인접 면에 누적하고 **정규화하지 않기**
  (길이가 world step → texel step 변환입니다). 프레임이 다르면 resolve되는 uv가 달라지고
  색이 달라집니다. 실측: 번들 경로로 다시 계산한 프레임과 run의 OBJ가 들고 있던 프레임의
  렌더 차이는 1/255 이하 (1/255를 넘는 픽셀 0개).
- **`unorm8` 번들은 8비트 UNORM입니다.** 계수 (k, 채널)마다 atlas coverage 안의
  min..max를 256단계로 나누고, offset/scale을 `config.json`에 싣습니다. 역양자화가 affine이라
  렌더러는 RGBA8 바이트를 그대로 올려 GPU가 bilinear 보간하게 두고, lookup 뒤에
  `offset + scale * sample`만 적용하면 됩니다 (보간과 순서가 바뀌어도 결과 동일, GPU 메모리는
  RGBA16F의 절반). boots(1024, 114뷰, 마스크 내부) 실측: sh.ktx2 2.28 MB (float16 8.97 MB),
  PSNR 22.89 vs 23.18 dB, SSIM 0.762 vs 0.799, LPIPS 0.267 vs 0.186. 범위가 순수 min..max라
  소수의 불안정한 텍셀이 전체 step을 넓힙니다 (boots에서 색 13-20레벨). 백분위 클리핑(0.1-99.9%)은
  4.17 MB에 23.14 dB / LPIPS 0.199로 손실이 훨씬 작습니다.
- **`float16` 번들은 계수가 float16이고, 저장 전에 두 번 깎습니다.** 둘 다 포맷을 바꾸지 않습니다 (여전히
  RGBA16F, GPU는 그대로 half float으로 샘플링). 목적은 압축기에게 찾을 거리를 주는 것뿐입니다.
  - *고차항의 0 근처 값을 정확히 0으로*: 참조 구현(`surface-light-fields-for-planar-paintings`의
    `PerPixelSH.save_sh_pngs`)과 같은 규칙으로, 계수 k는 `sh_zero_threshold_step * k` 미만이면
    0이 됩니다. 차수가 높을수록 시각적 기여가 작은데 임계값은 커지므로 더 많이 0이 됩니다.
    k=0(기본색)은 절대 건드리지 않습니다.
  - *mantissa 하위 비트 반올림*: float16의 mantissa 10비트 중 `sh_mantissa_bits`개만 남깁니다.
    **이쪽이 훨씬 큰 레버**입니다 - 0으로 누르고 남은 값들은 mantissa 바닥이 사실상 노이즈라,
    어떤 렌더러도 보여줄 수 없는 비트에서만 서로 다릅니다. 지수부 기준 상대 정밀도 절삭이라
    계수 크기와 무관하게 같은 비율을 깎습니다 (고정소수점이라면 최댓값에 맞춰야 합니다).

  run 4개 × 시점 3개에서 float32 피팅 대비 실측: 0 누르기만 하면 raw float32의 23-31% 크기에
  45-50 dB PSNR이고, 거기서 mantissa 비트를 하나 줄일 때마다 파일이 약 9%씩 더 줍니다.
  6비트까지는 품질 손실이 1.1 dB 미만 + 2/255 초과 픽셀 비율 변화 0.15%p 미만으로 사실상
  공짜이고, 기본값인 5비트는 최대 2.5 dB / 0.5%p입니다. 4비트에서 무너집니다 (오차 픽셀이
  갑자기 5-10%로 뜀). 그래서 기본값을 그 한 칸 위에 뒀습니다.
- **포맷은 KTX2 + zlib이지 Basis/ASTC가 아닙니다.** SH 계수와 height는 색이 아니라 수치라서
  8비트 블록 압축을 먹이면 march가 층 사이에서 밴딩합니다. 채널 수를 1/2/4로 제한하는 것도
  의도적입니다 - three.js `KTX2Loader`의 FORMAT_MAP에 평범한 3채널 float(R16G16B16_SFLOAT)이
  없어서(3채널 항목은 packed 32비트와 블록 압축뿐), RGB16F는 파싱은 되고 업로드할 포맷이
  없습니다. 그래서 alpha는 0 패딩으로 버립니다 (압축 후 약 +7%).
- **supercompression은 zstd가 아니라 ZLIB(스킴 3)입니다.** 압축률이 아니라 배포 선택입니다:
  브라우저는 `DecompressionStream("deflate")`로 zlib을 네이티브로 풀지만, zstd는 첫 텍스처가
  뜨기 전에 wasm 디코더를 받아서 초기화해야 합니다. scan6 SH 아틀라스 실측으로 zlib -9가
  8.96 MB, zstd -12가 8.06 MB이니 페이로드 11%와 모든 소비자의 wasm 의존성을 맞바꾼 셈입니다.
  **단, three.js `KTX2Loader`는 스킴 0과 2만 처리하고 이 스킴에는 예외를 던집니다** - 자체
  로더를 쓰거나, level 데이터를 먼저 풀어서 넘겨야 합니다.
- **`height.range` / `height.num_steps`는 튜닝값이 아니라 포맷의 일부**입니다. POM march가 훑는
  슬랩을 정의하므로, 다른 값을 쓰면 프래그먼트마다 다른 uv로 resolve되어 학습된 적 없는 텍셀의
  계수를 읽습니다. `num_steps`는 params가 아니라 slf stage의 manifest에서 읽습니다.
- **초기 카메라는 캡처가 찍힌 방향**입니다. 고정 azimuth는 평면 물체에서 뒷면을 보게 되므로
  (Holy Family에서 실제로 검은 화면), ingest 카메라들의 중심 방향 평균으로 정합니다. orbit
  파라미터와 함께 resolve된 eye/target/view matrix도 같이 실어 orbit 규약 없이도 재현됩니다.
- **`height.ktx2`의 g 채널은 baking의 아틀라스 커버리지**(차트 내부 + gutter)이지 SH 피팅의
  valid 마스크가 아닙니다. 커버리지 안이지만 관측되지 않은 텍셀은 이웃의 실제 피팅을
  nearest-copy로 들고 있어 정상적으로 셰이딩되며, 이걸 마스킹하면 표면에 구멍이 뚫립니다.

`ktx2.py` / `gltf.py`는 컨테이너를 직접 씁니다 (libktx는 리눅스 wheel이 없고, trimesh의 GLB
export는 회색 1x1 PNG 머티리얼을 끼워 넣습니다). 손으로 쓴 바이너리 포맷이라 자기 리더로
왕복하는 것만으로는 검증이 안 되므로, Khronos `ktx create`가 같은 텍스처에 내놓는 헤더/DFD 바이트와
직접 비교하고 GLB는 trimesh의 독립 importer로 읽어 확인했습니다.

번들만 보고 도는 뷰어가 `viewers/view_bundle.py`입니다. 온 run은 열지 않고, GLSL과 프레임
계산만 파이프라인과 공유합니다 (재현성이 목적이므로 의도적으로).

```bash
python viewers/view_bundle.py data/runs/<run_id>/10_export/<run_id>_unorm8.snap3d
```
