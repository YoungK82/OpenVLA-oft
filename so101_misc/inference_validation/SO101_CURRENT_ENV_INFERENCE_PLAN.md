**SO-101 25k 체크포인트: 현재 Linux 서버 + 기존 Windows 클라이언트 추론 계획**

후속 작업: 이 계획 이후 실제 GPU 추론·입력 안전성 검사를 완료하고 서버 검증을 보완했다. 최신 결과와 실행 명령은 [추론 검증 결과 및 Windows 연결 방법](/home/yhkim/openvla-oft/SO101_INFERENCE_VALIDATION.md)을 따른다. 아래의 미실행 상태와 기존 오류 동작은 최초 계획 시점의 기록이다.

점검일: 2026-09-28. 대상 저장소: `/home/yhkim/openvla-oft`, HEAD `f2ac6e8`에 기존 로컬 변경이 있는 상태.
사용자 확인 사항: **Windows 환경과 실제 제어에 사용했던 클라이언트 파일을 그대로 사용한다.**

현재 점검 결과, 체크포인트와 서버용 Conda 환경은 이미 준비되어 있다. 우선 필요한 작업은 기존 환경 선택, 현재 경로로 서버 실행, Windows 접속 경로 연결이다. 점검 범위에서는 모델 구조나 학습 코드를 반드시 수정해야 할 오류가 발견되지 않았다. 전체 모델 로딩과 `/act` 추론은 아래 실행 단계에서 검증해야 한다.

**1. 실제 확인한 상태**

| 항목 | 확인 결과 | 판단 |
| --- | --- | --- |
| 25k 체크포인트 | `checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt` | 이번 실행 대상 |
| 병합 모델 | safetensors 4개, index의 982개 tensor가 shard header와 일치하고 각 파일 길이도 일치 | 파일 누락·길이 잘림은 발견되지 않음. 전체 내용 checksum 검사는 수행하지 않음 |
| 별도 학습 모듈 | `action_head--25000_checkpoint.pt`, `proprio_projector--25000_checkpoint.pt` CPU 로드 성공 | head 출력 6D, projector 입력 6D 확인 |
| 전처리·통계 | tokenizer/processor 파일, `dataset_statistics.json` 존재 | 통계 키 `so101_block_into_cup_50_v5`; 현재 모델 코드와 checkpoint 모델 코드 일치 |
| GPU | RTX PRO 6000 Blackwell Server Edition 2장, 각 약 96 GiB; driver `595.58.03` | H200 환경 복사 대신 현재 설치된 GPU 환경 사용 |
| 서버 Python | `/home/yhkim/miniconda3/envs/openvla-oft/bin/python`, `3.10.21` | 기존 환경 재사용 |
| PyTorch | `2.11.0+cu130`, torchvision `0.26.0+cu130`, torchaudio `2.11.0+cu130` | `sm_120` 포함, GPU 0의 BF16 행렬곱 실제 통과 |
| Transformers | OFT fork `4.40.1`, commit `bc339d9ad707454c0c115970db43c260067c61ab` | 일반 Transformers로 교체하지 않고 유지 |
| 주요 의존성 | TensorFlow `2.15.0`, TFDS `4.9.3`, NumPy `1.26.4`, timm `0.9.10`, peft `0.11.1` | `pip check` 통과, `deploy.py --help` import/CLI 진입 성공 |
| Flash Attention | 별도 `flash-attn` 패키지 없음; 현재 로더도 강제 선택하지 않음 | 설치를 선행 조건으로 추가할 근거 없음 |
| 검증 데이터 | `/home/yhkim/datasets/rlds/so101_block_into_cup_50_v5/1.1.0` | train 45 / validation 5; 실제 validation 한 step 읽기 성공 |
| SO-101 설정 | 명시적 `--robot_platform so101` 선택 검사 통과 | chunk 30, action 6D, proprio 6D, bounds 정규화 |
| 서버 가동 상태 | 점검 당시 `deploy.py` 프로세스와 8777 listener 없음 | 추론 서버 실행 필요 |

검증 샘플은 episode 24이며, front/top/wrist 각각 `(256, 256, 3)` uint8, state `(6,)` float32였다. 지시는 `Pick up the block and place it into the cup.`이다. 모델 입력 전처리가 이미지를 224 크기로 맞춘다.

체크포인트의 `lora_adapter`에는 이전 `/workspace/models/...` 경로가 남아 있지만, 현재 배포 로더는 체크포인트 루트의 병합 모델을 읽는다. 이 배포 경로에서 별도 base model 다운로드나 LoRA 재병합은 필요하지 않다.

**2. 지금 처리할 문제와 수정 범위**

| 우선순위 | 근거와 영향 | 처리 |
| --- | --- | --- |
| 필수 | 현재 점검 셸의 기본 `python`은 Conda base의 `3.14.6`이며 torch가 없다. `serve_portable.sh`는 PATH의 `python`을 두 번 사용한다. | `openvla-oft` 활성화 후 실행. 활성화 없이 실행하면 모델 로딩 전 `import torch`에서 실패할 환경이다. |
| 필수 | `SO101_PORTABLE_INFERENCE.md`는 `~/openvla-oft/code/OpenVLA-oft`와 새 환경 `oft-infer`를 예시로 사용한다. | 현재 경로 `/home/yhkim/openvla-oft`와 기존 환경 `openvla-oft`로 실행 명령을 정정한다. 해당 문서의 설치 절차를 다시 실행할 필요는 없다. |
| 필수 | `serve_portable.sh` 기본 bind는 `127.0.0.1:8777`이다. | 아래 SSH 터널을 통해 Windows에서 연결한다. 직접 접속을 택하면 `VLA_HOST=0.0.0.0`과 실제 서버 IP를 사용한다. |
| 실행 전 확인 | 점검 순간 GPU 0·1 모두 다른 작업으로 utilization 100%였다. | 기존 작업 종료 후 사용 가능한 GPU를 선택한다. 현재 VRAM 여유만으로 응답 속도를 판단하지 않는다. |
| 유지 | 과거 Windows 실행은 query당 30개 action을 실행했다는 기록이 있다. 현재 저장소 클라이언트는 실제 제어에서 3개 초과를 거부한다. | 기존 Windows 파일·환경·실행 설정을 보존한다. 이번 서버 이식 때문에 저장소 클라이언트를 덮어쓰지 않는다. |
| 선택 | 실행할 Python을 스크립트 안에서도 명확히 지정하면 환경 실수를 줄일 수 있다. | 후속 코드 수정 시 `python_bin="${VLA_PYTHON:-python}"`을 두고 preflight와 `exec` 모두 `"$python_bin"` 사용. 기본값은 유지한다. 아래 명령은 이 수정 없이 실행 가능하다. |

근거 코드: [실행 스크립트](/home/yhkim/openvla-oft/experiments/robot/so101/serve_portable.sh:13), [모델 로더](/home/yhkim/openvla-oft/experiments/robot/openvla_utils.py:273), [현재 클라이언트 실행 제한](/home/yhkim/openvla-oft/experiments/robot/so101/run_so101_eval.py:121), [과거 Windows 실행 기록](/home/yhkim/openvla-oft/SO101.md:393).

이 계획의 초기 실행에는 기존 서버 코드와 `serve_portable.sh`를 사용한다. 새 서빙 프레임워크, Kubernetes 구성, 재학습, 양자화, Windows 패키지 재설치는 필요 작업에 포함하지 않는다.

**3. 서버 실행 — Linux 터미널 A**

아래 명령은 후속 실행용이며 이번 점검에서 서버를 기동한 것은 아니다. `CUDA_VISIBLE_DEVICES=0`은 GPU 0을 사용할 수 있을 때의 예시다. GPU 1을 배정하면 `1`로 바꾼다. 코드의 `cuda:0`은 선택된 GPU에 대응한다.

```bash
source /home/yhkim/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/yhkim/openvla-oft
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export VLA_HOST=127.0.0.1
export VLA_PORT=8777
mkdir -p outputs/so101-inference-current
export VLA_DIAGNOSTIC_LOG="$PWD/outputs/so101-inference-current/actions.jsonl"
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
set -o pipefail
bash experiments/robot/so101/serve_portable.sh \
  "$PWD/checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt" \
  2>&1 | tee outputs/so101-inference-current/server.log
```

실행 스크립트가 이미 고정하는 옵션은 그대로 사용한다.

| 설정 | 값 |
| --- | --- |
| robot platform | `so101` |
| action head | L1 regression; diffusion / FiLM 비활성화 |
| 관측 | front → top → wrist 3장 + proprio 6D + instruction |
| center crop | `True` |
| unnorm key | `so101_block_into_cup_50_v5` |
| 출력 | 역정규화된 절대 위치 action `(30, 6)` |

통과 기준은 SO101 상수 출력, 모델 4개 shard와 head/projector 로딩 완료, Uvicorn의 `127.0.0.1:8777` startup이다. 서버 로더는 checkpoint config를 백업·갱신하고 필요한 모델 코드 파일을 동기화하므로 해당 디렉터리의 쓰기 권한이 필요하다. 현재 파일 소유자는 `yhkim`이다.

**4. 기록 데이터로 한 번 추론 — Linux 터미널 B**

```bash
source /home/yhkim/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/yhkim/openvla-oft
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail
CUDA_VISIBLE_DEVICES="" python experiments/robot/so101/smoke_test_server.py \
  --data-root /home/yhkim/datasets/rlds \
  --dataset-name so101_block_into_cup_50_v5 \
  --server-endpoint http://127.0.0.1:8777/act \
  --timeout 120 \
  2>&1 | tee outputs/so101-inference-current/smoke.log
```

이 명령의 `CUDA_VISIBLE_DEVICES=""`는 데이터 읽기 프로세스에만 적용된다. 터미널 A의 GPU 추론에는 영향을 주지 않는다. 현재 검사 도구는 데이터 버전 디렉터리 이름을 정렬해 마지막 것을 선택하므로 출력의 `Builder`가 `1.1.0`인지 확인한다.

통과 기준은 `Output actions: shape=(30, 6)`과 `SO-101 checkpoint server smoke test passed.` 출력이다. 도구가 NaN/Inf도 검사한다. 서버는 예외 시 HTTP 200과 `"error"`를 반환할 수 있으므로 HTTP status만으로 성공 처리하지 않는다.

첫 요청이 통과하면 같은 요청을 한 번 더 보내 준비 이후 응답도 확인한다. `actions.jsonl`의 checkpoint 경로와 `inference_seconds`, 전체 왕복시간을 함께 기록한다. 현재 저장소의 HTTP helper timeout은 30초이며 기존 Windows 파일의 실제 timeout은 따로 확인해야 한다. 30 Hz action 실행 설정이 모델 요청까지 30 Hz라는 의미는 아니다.

이미 있는 `outputs/so101-local-rlds-comparison.json`에는 validation 한 episode의 세 frame 결과가 있다. 첫 action 평균 MAE 약 2.068, chunk 평균 MAE 약 3.230이지만 checkpoint·패키지·GPU 식별 정보가 없어 이번 25k 환경의 성공 증거로 대신 사용할 수 없다. 또한 관절과 gripper 단위가 섞인 평균이므로 성공률이나 순수한 각도 오차로 해석하지 않는다.

**5. 기존 Windows 클라이언트 연결**

기본 연결은 Windows → SSH 터널 → 현재 Linux의 `127.0.0.1:8777`이다. 현재 서버로 SSH 접속할 수 있다는 전제이며, `<현재_Linux_서버_IP>`는 Windows에서 접속 가능한 실제 IP 또는 SSH host 이름으로 교체한다.

Windows PowerShell의 별도 창에서 실행하고 유지한다.

```powershell
ssh -p 50123 -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8777:127.0.0.1:8777 yhkim@<현재_Linux_서버_IP>
```

다른 PowerShell 창에서 서버 도달 여부를 확인한다.

```powershell
curl.exe --fail http://127.0.0.1:8777/openapi.json
```

응답의 `/act`는 네트워크와 서버 route 확인용이다. 모델 출력 검증은 앞 단계와 실제 client 요청으로 수행한다. Windows의 8777이 이미 사용 중이면 터널 왼쪽 port와 클라이언트 port만 함께 다른 값으로 바꾼다.

기존 Windows 실행 명령에서 서버 접속 옵션을 다음으로 맞춘다.

```text
--vla_server_url 127.0.0.1 --vla_server_port 8777
```

현재 저장소의 URL 처리에서는 `http://...` 형식을 쓰면 별도 port 옵션을 붙이지 않는다. 따라서 위처럼 host와 port를 분리하거나, URL을 `http://127.0.0.1:8777/act`까지 완전하게 지정한다. 기존 Windows 코드의 처리도 확인한다.

직접 LAN 연결을 원하면 터미널 A에서 `VLA_HOST=0.0.0.0`으로 실행하고 Windows의 host를 현재 Linux IP로 바꾼다. 이 경우 서버의 TCP 8777 접근이 허용되어 있어야 한다. 터널 연결은 서버 bind 변경이 필요 없다.

기존 Windows COM port, robot id, calibration, action 제한, task, 실행 chunk 수는 이전 설정을 기준으로 한다. 기록상 마지막 카메라 매핑은 `front=2`, `top=1`, `wrist=0`이지만 장치 번호는 재연결로 바뀔 수 있다. 실제 뷰와 모델의 front → top → wrist 순서를 맞춘다. payload는 아래 계약이어야 한다.

```text
full_image: front RGB uint8 image
additional_images: [top RGB uint8 image, wrist RGB uint8 image]
state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
instruction: Pick up the block and place it into the cup.
response: 30 × 6 absolute targets
```

팔 관절은 기존 degree 설정을 유지하고 gripper는 기존 LeRobot calibration/단위를 유지한다. 여섯 값을 모두 각도로 취급하거나 절대 action을 delta로 바꾸지 않는다.

먼저 기존 client의 `--dry_run True`로 짧게 요청하여 실제 세 카메라와 state가 들어가고 action이 반환되는지 확인한다. 현재 저장소의 dry-run도 robot 연결·관측은 수행하고 `send_action`만 생략하므로, Windows의 기존 파일도 같은 동작인지 확인한다. 그다음 기존에 사용하던 실제 제어 명령으로 짧은 rollout을 확인한다. query당 30개 action을 쓰던 파일을 이번 checkout으로 교체하면 제한에 걸리므로, 파일 보존이 필요하다.

**6. 실패했을 때 바꿀 부분과 완료 조건**

| 현상 | 먼저 볼 부분 | 처리 방향 |
| --- | --- | --- |
| `No module named torch` | `sys.executable` | Conda 환경 선택 수정 |
| CUDA/모델 로딩 traceback | 서버 로그, 선택 GPU, 현재 점유 | 발생한 연산과 패키지를 확인한 뒤 해당 부분만 수정. 지금의 BF16 작은 연산 성공이 전체 모델 forward 성공까지 보장하지는 않음 |
| action head / projector shape mismatch | `Using SO101 constants`, 25k component 파일 | 30/6/6 및 동일 checkpoint의 head/projector 확인 |
| Windows 연결 실패 | Linux local smoke 결과, tunnel, URL/port | 접속 경로 수정 |
| HTTP 200이지만 `error` 또는 잘못된 shape | 서버 traceback, 세 이미지와 6D state | payload 계약 또는 실제 inference 오류 수정 |
| 유효한 action이지만 움직임이 다름 | 기존 Windows 파일·카메라·calibration·chunk 수·응답시간 | 이전 실행과의 차이를 먼저 좁힌 뒤 필요한 부분만 수정 |

이번 CPU 검사에서 TensorFlow의 CUDA factory/cuInit 로그가 나왔지만 CLI와 데이터 읽기는 종료 코드 0으로 완료됐다. CPU 검사에는 GPU를 의도적으로 숨겼다. 이 로그만으로 현재 PyTorch CUDA 환경이 고장났다고 판단하지 않는다. checkpoint의 Transformers 버전 표기 `4.44.2`와 설치된 fork `4.40.1`의 차이도, import 실패 근거 없이 일반 Transformers 교체 사유로 삼지 않는다.

완료는 세 단계로 판정한다: **현재 서버에서 25k 요청 → 유한한 `(30, 6)` action**, **기존 Windows client에서 같은 서버 요청 성공**, **기존 카메라·calibration·제어 설정으로 실제 로봇 rollout 확인**. 기록 데이터의 action 출력만으로 물리 작업 성공을 단정하지 않는다.

이번 점검에서는 shard 구조, CPU component 로딩, 작은 GPU BF16 연산, 의존성, CLI import, RLDS 한 step 읽기까지 확인했다. 전체 모델 GPU 로딩, `/act` 요청, Windows 접속 및 로봇 제어는 실행하지 않았다. 이번 변경은 이 계획 문서 추가뿐이며 기존 코드·환경·체크포인트는 수정하지 않았다.

참고한 [공식 LIBERO 문서](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)는 OFT fork와 학습 augmentation에 맞춘 center crop의 근거다. SO-101의 이미지 수와 action 차원은 이 저장소의 실제 학습·배포 계약을 따른다. Blackwell용 PyTorch 지원은 [공식 PyTorch 2.7 안내](https://pytorch.org/blog/pytorch-2-7/)에서도 확인되며, 이번 환경 선택은 현재 설치된 `2.11.0+cu130`의 실제 GPU 연산 결과를 기준으로 했다.
