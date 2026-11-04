# shipbuilding: 마그넷 크레인 OpenVLA-OFT

조선소 마그넷 크레인 데이터(강판 들기·옮기기·내려놓기)로 OpenVLA-OFT를 파인튜닝해 **마그넷 10개의 on/off**를 예측한다.
모델 구조(LoRA + FiLM + proprio projector + L1 회귀 head)는 원본 OpenVLA-OFT 그대로이고, **입력과 action space만 바뀌었다.**

## 어느 폴더를 쓰나

**`b4_state/`를 쓴다** (2026-11-04 기준 현재 설계).

| 폴더 | 지시문 | proprio | 마그넷 세기 | 데이터셋 이름 | split |
|---|---|---|---|---|---|
| **`b4_state/`** | task만 | `observation.state` 그대로 [하중, x, y, z] (4) | **사용 안 함** | `crane_b_magnet_onoff_state_tvt` | train 50 / val 10 / test 10 |
| `b1_with_power/` (이전) | task + 하중 + 세기 | x, y, z (3) | 언어 | `crane_b_magnet_onoff_loadpower` | train 60 / val 10 |
| `b2_state_power/` (이전) | task + 세기 | 하중, x, y, z (4) | 언어 | `crane_b_magnet_onoff_state_power` | train 60 / val 10 |
| `b3_proprio_power/` (이전) | task만 | x, y, z, 하중, 세기 (5) | proprio | `crane_b_magnet_onoff_proprio_power` | train 60 / val 10 |

- b1·b2·b3는 마그넷 세기를 입력으로 쓴 이전 실험이다. 세기가 on/off 정답을 거의 그대로 알려 주고, test split도 없어서 b4로 정리했다.
- 각 폴더는 독립된 사본이다(자체 `prismatic/`, `vla-scripts/`). **반드시 그 폴더 안에서 `export PYTHONPATH=$PWD` 후 실행한다.**
  conda 환경 `openvla-oft`의 editable 설치는 `../so101/`을 가리키므로, 빼먹으면 다른 폴더의 코드가 돈다.
- 레포 루트는 원본 `moojink/openvla-oft`와 같게 유지하는 곳이고, `so101/`은 별개 작업이다. 둘 다 크레인에 쓰지 않는다.
- transformers는 `moojink/transformers-openvla-oft` fork가 설치돼 있어야 한다(bidirectional attention).

## 원본 OpenVLA-OFT와 다른 점

| | 원본 (ALOHA 예시) | 크레인 (b4) |
|---|---|---|
| action | 관절 14차원, 연속값 | 마그넷 10개 on/off (0/1 → -1/1 정규화), chunk 8 (약 0.27초) |
| 카메라 | 3대 (high + 손목 2) | exo 3대 (exo_1, exo, exo_3 → primary/secondary/tertiary) |
| proprio | 관절 상태 (qpos) | `observation.state` [하중, x 주행, y 횡행, z 권상] |
| 지시문 | 고정 task | 에피소드마다 다른 task (강판 번호·적치 위치) → FiLM 사용 |
| 모델 | LoRA + FiLM + L1 head | 같음 |

## 흐름: 변환 → 학습 → 평가

### 1. 변환 (크레인 전용으로 가장 많이 새로 만든 부분)
- `experiments/robot/crane/convert_crane_to_rlds.py`
  - LeRobot v3 parquet과 mp4를 RLDS(TFDS)로 바꾼다.
  - 카메라 매핑과 256×256 리사이즈(모델 입력은 224), 지시문·state 저장을 여기서 한다.
  - 불량 에피소드 제외(`EXCLUDED_TIDS`)와 train/val/test split(`VALIDATION_PER_DATE`, `TEST_PER_DATE`, seed 7)도 여기서 정한다.
  - 제외한 5개(75 → 70): 마그넷 ON이 1프레임뿐인 2개, 짧고 하중이 한 번도 안 잡힌 2개(강판을 실제로 안 듦),
    녹화가 중간에 잘린 1개. TID별 사유는 `constants.py` 주석에 있다. val은 제외 전 75개에서 먼저 뽑아 제외와 무관하게 같다.
  - parquet에서 쓰는 값은 `action`(마그넷 10개 on/off)과 `observation.state` 둘뿐이다.
    `observation.magnet_power_level`(마그넷 세기)은 읽지 않고, `current`(전류)는 전부 NaN이라 쓰지 않는다.
- `experiments/robot/crane/crane_utils.py`: 지시문 생성 함수(b4는 task 그대로). 변환과 추론이 같은 함수를 쓴다.
- `experiments/robot/crane/constants.py`: 데이터셋 이름, 카메라, split 설정.
- `experiments/robot/crane/verify_rlds_dataset.py`: 변환 결과 전수 검사(영상 크기, action 0/1, state, 지시문, 세기 미포함,
  split이 manifest와 같은지).

### 2. 학습 (원본 `finetune.py` + 데이터 등록 몇 줄)
- 데이터 등록:
  - `prismatic/vla/constants.py`: `CRANE_CONSTANTS`(ACTION_DIM 10, chunk 8, PROPRIO_DIM 4)
  - `prismatic/vla/datasets/rlds/oxe/configs.py`, `transforms.py`, `mixtures.py`, `materialize.py`: 이름에 `crane`이 들어간 데이터셋은 자동으로 크레인 설정을 쓴다.
- `prismatic/vla/datasets/datasets.py`: 카메라 3대를 로드한다. **이게 빠지면 에러 없이 카메라 1대로만 학습된다.**
- `vla-scripts/finetune.py`: 크레인 인식 외에 SO-101 작업 때 넣은 학습 편의 기능이 있다(크레인 전용 아님).
  - 정확한 이어서 학습(`--resume_from`)
  - `STOP` 파일로 저장 후 종료
  - 자동 LR 감소(`--auto_lr_decay`)
- `experiments/robot/crane/train_crane.sh`: aga01용 실행 스크립트. Hawk(SLURM)는 `slurm_train_crane.sbatch`.
  - 유효 batch 8, 25K 스텝, LR 5e-4 고정이 기본값이다.

### 3. 평가
- `vla-scripts/merge_lora_weights_and_save.py`(원본): LoRA 병합. 체크포인트는 병합 안 된 상태로 저장된다.
- `experiments/robot/crane/serve_crane.sh`: `deploy.py`로 정책 서버를 띄운다(카메라 3대, FiLM, proprio).
- `experiments/robot/crane/eval_server_on_rlds.py`: RLDS 프레임을 서버에 보내 채점한다.
  - 지표: 마그넷별 정확도(0.5 기준), 10개 모두 맞춘 비율, on/off 시점 오차.
- 체크포인트는 `--split validation`으로 고르고, `--split test`는 마지막에 한 번만 본다.

## 명령 (`b4_state/` 안에서)

```bash
cd shipbuilding/b4_state
export PYTHONPATH=$PWD
PY=~/miniconda3/envs/openvla-oft/bin/python

# 1) 변환: CPU, 약 1.5~2시간 → tmux에서
CUDA_VISIBLE_DEVICES= $PY experiments/robot/crane/convert_crane_to_rlds.py \
  --source-root /home/shared/2026shipbuildingAX/Data_201012-201014 --output-root ~/datasets/rlds
$PY experiments/robot/crane/verify_rlds_dataset.py --data-root ~/datasets/rlds

# 2) 학습: tmux에서. 끊기면 같은 명령을 다시 실행하면 정확히 이어서 학습한다
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/datasets/rlds BASE_MODEL=<openvla-7b 사본> \
  bash experiments/robot/crane/train_crane.sh
#   GPU 2장: CUDA_VISIBLE_DEVICES=0,1 NPROC=2 BATCH_SIZE=2 GRAD_ACCUM=2 ... (유효 batch 8)
#   예정된 정지: touch checkpoints/crane_b_magnet_onoff_state_tvt/STOP  또는  STOP_AT=HH:MM

# 3) 평가
CKPT=checkpoints/crane_b_magnet_onoff_state_tvt--20000_chkpt
$PY vla-scripts/merge_lora_weights_and_save.py --base_checkpoint <openvla-7b> --lora_finetuned_checkpoint_dir $CKPT
CUDA_VISIBLE_DEVICES=0 VLA_PORT=8777 bash experiments/robot/crane/serve_crane.sh $CKPT
$PY experiments/robot/crane/eval_server_on_rlds.py --data-root ~/datasets/rlds --split validation  # 체크포인트 고르기
$PY experiments/robot/crane/eval_server_on_rlds.py --data-root ~/datasets/rlds --split test        # 마지막에 1회
```

- 체크포인트는 `checkpoints/crane_b_magnet_onoff_state_tvt--<스텝>_chkpt`에 저장된다(5K 간격).
- 이어서 학습할 상태는 `checkpoints/crane_b_magnet_onoff_state_tvt--resume_latest`에 250스텝마다 갱신된다.

## 주의
- **`BASE_MODEL`은 공유 폴더(`/home/shared/openvla-oft/models/openvla-7b-47a0ec7`)를 직접 쓰지 말고 사본을 쓴다.**
  `finetune.py`가 `config.json`을 고쳐 쓰기 때문이다. safetensors만 심볼릭 링크하고 나머지는 복사한 폴더를 쓴다.
- 공용 서버(aga01)에서는 GPU를 쓰기 전에 다른 사용자의 사용량을 먼저 확인한다(`nvidia-smi`).
  batch 8을 한 장에 올리면 96GB에서도 OOM이 난다.
- 데이터, RLDS, 체크포인트는 커밋하지 않는다(public 레포).
- **남아 있는 정답 단서:** state는 실시간 센서가 아니라 ERP 이벤트 값을 유지한 계단형 값이다.
  하중은 마그넷을 켜기 직전에 강판 무게로 뛰고, x/y/z는 시작 5초 동안 0이다. 점수가 좋아도 영상보다 이 값을 따라간 것일 수 있다.
- 이전 b1·b2·b3 체크포인트는 b4 test 에피소드로 학습됐으므로 test 평가에 쓰면 안 된다.
- 자세한 데이터 분석, 결정 이력, 실측 메모리·속도는 `CLAUDE.md`에 있다.
