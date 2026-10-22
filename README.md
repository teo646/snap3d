# snap3d

여러 장의 사진에서 3D 물체를 만들어, 웹(WebGL)에서 바로 렌더링할 수 있는 **mesh + SH 텍스처**로 내보내는 파이프라인입니다.

## 기술 소개

물체를 여러 방향에서 찍은 사진(단일 물체, 고정 조명)을 넣으면 다음 순서로 처리합니다.

```
사진 ─ SfM(COLMAP) ─ 마스킹(SAM3) ─ 표면 복원(Fast-PGSR) ─ 구멍 메우기 ─ 단순화 ─ UV 펼치기 ─ 베이킹 ─ SLF 피팅 ─ 내보내기
```

| 단계 | 하는 일 | 사용 기술 |
|---|---|---|
| ingest | 카메라 포즈와 sparse 점 구름 | COLMAP |
| resize_dataset | 학습 해상도의 핀홀 이미지로 변환 | |
| masking | 사진마다 물체 마스크 (텍스트 프롬프트) | SAM3 |
| surface_recon | 고해상도 메시 | Fast-PGSR (가우시안 스플래팅 → TSDF 융합) |
| hole_close, decimation | 구멍을 막고 약 5만 면으로 단순화 | PyMeshLab |
| uv_unwrap | UV 아틀라스 | xatlas |
| baking | 고해상도 → 저해상도 높이맵 | 레이캐스팅 |
| slf | **텍셀마다 구면조화(SH) 계수를 피팅** | 공액경사법 + 정규방정식 |
| export | 렌더용 번들 `mesh.glb`, `sh.ktx2`, `height.ktx2` | glTF / KTX2 |

핵심은 SLF(Surface Light Field)입니다. 텍셀마다 색을 하나 저장하는 대신 **보는 방향에 따라 달라지는 색**을 SH로 저장해서, 하이라이트와 반사처럼 시점에 따라 변하는 모습을 사진에서 학습합니다. 결과는 가벼운 저폴리 메시와 텍스처 두 장이라서 별도 신경망 없이 셰이더만으로 렌더링할 수 있습니다.

선택 기능으로, 광택 있는 물체를 위한 하이라이트 제거(SHIQ)와 형상 보정을 위한 단안 깊이 prior(Metric3D)가 있습니다.

## 환경 설정

Linux, NVIDIA GPU(RTX 50 시리즈 기준으로 확인), [conda](https://docs.conda.io), git이 필요합니다.

```bash
git clone --recursive https://github.com/teo646/snap3d.git
cd snap3d
./setup.sh
```

`setup.sh`가 하는 일은 다음과 같습니다.

1. `sh-texture` 환경 생성: Python 3.12, GPU COLMAP, 그리고 `pip install -e ".[all]"`
2. 서브모듈(`third_party/FastGS`, `third_party/Metric3D`)을 받고 패치 적용
3. `fastgs` 환경 생성: torch 2.11(CUDA 12.8). FastGS가 자체 CUDA 확장을 컴파일해서 쓰기 때문에 본체와 환경이 분리되어 있습니다.
4. FastGS CUDA 확장 4개 컴파일 (CUDA 12.8 툴킷이 없으면 환경 안에 설치)

실행하는 환경은 이 두 개뿐입니다. 파이프라인은 필요할 때 `fastgs` 환경을 알아서 호출하니 `sh-texture`만 활성화하면 됩니다. `./setup.sh --skip-fastgs`는 본체만 만듭니다(표면 복원은 못 합니다).

마지막으로 SAM3 체크포인트가 접근 권한이 있어야 받을 수 있습니다. [facebook/sam3](https://huggingface.co/facebook/sam3)에서 접근을 요청한 뒤 로그인하세요.

```bash
conda activate sh-texture
hf auth login
```

별도 환경이 이름과 위치를 못 찾으면 `SNAP3D_FASTGS_PYTHON=/path/to/python`처럼 환경 변수로 지정할 수 있습니다(`SNAP3D_DA3_PYTHON`, `SNAP3D_SHIQ_PYTHON`도 같습니다).

## 실행

```bash
conda activate sh-texture
python -m pipeline.orchestrator.runner \
    --config pipeline/orchestrator/configs/photo_pipeline.yaml \
    --input /path/to/photos \
    --text "물체를 설명하는 영어 단어"
```

- `--input`: 사진이 든 폴더
- `--text`: SAM3가 분할할 물체의 설명(예: `apple`, `teacup`). 못 찾으면 다른 단어로 바꿔 보세요. 생략하면 마스킹을 건너뜁니다.
- `--until <단계>`: 그 단계까지만 실행, `--run-id 이름`: 결과 폴더 이름, `--force`: 캐시 무시

결과는 `data/runs/<run_id>/`에 단계별 폴더로 쌓이고, 입력이 바뀌지 않은 단계는 다음 실행에서 건너뜁니다. 최종 산출물은 `13_export/` 안의 번들입니다(`..._unorm8.snap3d`는 작은 용량, `..._float16.snap3d`는 높은 정밀도).

결과 보기:

```bash
python viewers/view_slf.py data/runs/<run_id>                    # 학습된 SH 텍스처
python viewers/view_bundle.py data/runs/<run_id>/*_export/<run_id>_unorm8.snap3d   # 내보낸 번들만으로 렌더
python viewers/view_uv_unwrap.py data/runs/<run_id>              # UV 펼침 확인
```

### 광택이 있는 물체

`photo_pipeline_specular.yaml`은 하이라이트 제거(SHIQ)와 깊이 prior(Metric3D, 가중치 5)를 켠 설정입니다. 사용법은 같고 `--config`만 바꿉니다. 추가로 필요한 것:

- `third_party/SHIQ`를 clone하고([KosRud/SHIQ](https://github.com/KosRud/SHIQ)), 그 모델이 요구하는 별도 `shiq` 환경(Python 3.6, torch 1.5, CPU 실행)을 준비합니다.
- 깊이 prior의 가중치는 `--lambda-depth-prior W`로 바꿀 수 있고, 0이면 끕니다.
