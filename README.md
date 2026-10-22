# snap3d

여러 장의 사진에서 3D 물체를 만들어, 웹(WebGL)에서 바로 렌더링할 수 있는 mesh + SH 텍스처로 내보내는 파이프라인입니다.
가우시안 스플래팅 기반 표면 복원(Fast-PGSR)으로 메시를 만들고, 약 5만 면으로 단순화해 UV를 펼칩니다.
텍셀마다 구면조화(SH) 계수를 피팅해 보는 방향에 따라 달라지는 색(하이라이트, 반사)을 저장합니다.
결과는 저폴리 메시와 텍스처 두 장이라서 신경망 없이 셰이더만으로 렌더링할 수 있습니다.

## 실행

```bash
git clone --recursive https://github.com/teo646/snap3d.git
cd snap3d
./setup.sh                 # conda 환경 sh-texture, fastgs 생성 + CUDA 확장 빌드
conda activate sh-texture
hf auth login              # SAM3 접근 권한 필요 (huggingface.co/facebook/sam3)

python -m pipeline.orchestrator.runner \
    --config pipeline/orchestrator/configs/photo_pipeline.yaml \
    --input /path/to/photos \
    --text "물체를 설명하는 영어 단어"
```

결과는 `data/runs/<run_id>/`에 단계별 폴더로 쌓이고, 최종 번들은 `13_export/`에 생깁니다.

## 파이프라인

```
사진 ─ SfM ─ 마스킹 ─ 표면 복원 ─ 구멍 메우기 ─ 단순화 ─ UV 펼치기 ─ 베이킹 ─ SLF 피팅 ─ 내보내기
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
| slf | 텍셀마다 SH 계수 피팅 | 공액경사법 + 정규방정식 |
| export | 렌더용 번들 `mesh.glb`, `sh.ktx2`, `height.ktx2` | glTF / KTX2 |
