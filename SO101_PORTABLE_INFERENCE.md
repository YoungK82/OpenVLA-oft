# 일반 GPU 서버에서 SO-101 추론 실행

전송 대상은 이 코드 디렉터리와 아래 체크포인트 디렉터리 두 개 전체입니다.

- so101-b8-s25000-trimmed-job-1--25000_chkpt
- so101-b8-s25000-trimmed-job-1--15000_chkpt

모델 shard, action head, proprio projector, tokenizer, processor, 통계 파일을 함께 보존합니다.
병합된 모델이므로 추론에는 별도 기본 모델 다운로드가 필요 없습니다.
기존 환경(envs), 학습 데이터, Kubernetes 설정은 이번 전송에 필요 없습니다.
새 서버의 기존 데이터는 검증 요청을 만드는 데 사용할 수 있습니다.

## 1. 새 서버 설치

아래는 Linux, Conda, NVIDIA GPU를 가정한 설치 후보입니다. 이 서버에서는 GPU 실행을
검증하지 못했습니다. 새 서버에서 `nvidia-smi`로 GPU와 드라이버를 먼저 확인하세요.
CUDA 12.8 wheel을 지원하는 드라이버가 필요합니다. 기존 CUDA 12.1 환경을 복사하지 마세요.
PyTorch 설치 출처: https://pytorch.org/get-started/previous-versions/

```bash
conda create -n oft-infer python=3.10 -y
conda activate oft-infer
export PYTHONNOUSERSITE=1
cd ~/openvla-oft/code/OpenVLA-oft
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
```

프로젝트 설치에는 인터넷과 git이 필요합니다. 전용 Transformers fork도 pyproject.toml에
따라 설치됩니다. 일반 Transformers 패키지로 임의 대체하지 마세요.
현재 추론 코드가 TensorFlow 및 데이터 유틸리티도 import하므로 프로젝트 의존성을 설치합니다.
Flash Attention은 추론 로더에서 강제로 선택하지 않으므로 최초 설치에서 별도 설치하지 않습니다.
설치나 CUDA 확인이 실패하면 그 오류를 해결한 뒤 다음 단계로 진행하세요.

## 2. 최종 체크포인트 서버 실행 (터미널 A)

```bash
conda activate oft-infer
bash ~/openvla-oft/code/OpenVLA-oft/experiments/robot/so101/serve_portable.sh \
  ~/openvla-oft/checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt
```

경로가 다르면 위 두 경로를 실제 위치로 바꾸면 됩니다. 스크립트가 저장소 경로를 자동으로 찾고,
체크포인트 필수 파일과 CUDA 연산을 확인한 뒤 SO-101 옵션으로 실행합니다.
현재 로더는 체크포인트 config와 모델 코드 파일을 백업·갱신하므로 폴더 쓰기 권한이 필요합니다.
서버는 기본적으로 `127.0.0.1:8777`의 `/act`에서 요청을 받습니다.
서버가 준비됐다는 Uvicorn 로그를 확인하고 다음 단계로 진행하세요.
이 명령은 서버를 띄우며, 추론 결과는 요청을 보낸 뒤 생성됩니다. W&B 로그인은 필요 없습니다.

## 3. 데이터 한 샘플로 검증 (새 서버의 터미널 B)

기존 데이터가 준비된 RLDS/TFDS 형식이고 다음 구조일 때 사용할 수 있습니다.
`RLDS_ROOT/so101_block_into_cup_50_v5/VERSION/dataset_info.json` 및 TFRecord 파일들.
이미지 필드 image/top_image/wrist_image, 6D state, language_instruction 및 val 또는 validation
split이 필요합니다. LeRobot 원본만 있다면 이 검사 전에 RLDS 변환 또는 별도 요청 생성이 필요합니다.

```bash
conda activate oft-infer
export PYTHONNOUSERSITE=1
cd ~/openvla-oft/code/OpenVLA-oft
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="" python experiments/robot/so101/smoke_test_server.py \
  --data-root /실제/RLDS_ROOT \
  --dataset-name so101_block_into_cup_50_v5 \
  --server-endpoint http://127.0.0.1:8777/act \
  --timeout 120
```

클라이언트의 데이터 읽기는 CPU에서 수행하고, 터미널 A의 모델이 GPU에서 추론합니다.
`Output actions: shape=(30, 6)` 및 `SO-101 checkpoint server smoke test passed.`가 통과 기준입니다.
실제 로봇을 움직이지 않습니다. 이 검사는 출력 형식을 확인하며 실제 작업 성공률을 측정하지 않습니다.

## 4. 중간 체크포인트 검사

터미널 A에서 Ctrl+C로 서버를 종료한 뒤, 실행 명령의 `25000`을 `15000`으로 바꾸고
터미널 B의 검증을 반복합니다. 처음에는 한 번에 모델 하나만 로드합니다.

## 이후 학습

새 서버의 데이터 형식과 버전을 확인한 뒤 학습 경로를 지정합니다.
기본 모델부터 다시 학습하려면 기본 모델을 별도로 준비합니다.
현재 체크포인트에는 optimizer/scheduler 상태가 저장되지 않으므로 추가 학습과 정확한 재개는 구분합니다.
