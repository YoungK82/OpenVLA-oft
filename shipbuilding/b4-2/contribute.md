# b4-2: LR 2e-4 재학습 코드와 결과

`b4-2/`는 `b4-1/`(결정적 validation) 코드 사본에 학습 기록·재시작·평가 쪽 수정을 더한 폴더다. **b4-2의 최종 학습 코드**이며,
여기서 LR 2e-4로 처음부터 학습한 run(`crane_b_magnet_onoff_state_tvt--lr2e-4`)의 설정과 결과를 정리한다.
데이터셋은 b4와 같다(`crane_b_magnet_onoff_state_tvt`, train 50 / val 10 / test 10, 지시문 = task만, proprio = state `[하중, x, y, z]`).
모델 구조와 train loss(L1 회귀)는 원본 OpenVLA-OFT와 같다. 학습 코드를 다시 바꾸는 다음 실험은 `b4-3/`에서 한다.

- 이전 단계: `../b4-1/contribute.md` (validation을 고정 프레임 평가로 바꾼 이유, 원래 b4가 LR 5e-4에서 14K에 붕괴한 분석)
- 체크포인트: `b4-2/checkpoints/crane_b_magnet_onoff_state_tvt--lr2e-4--{1000,...,14000}_chkpt` (1K 간격, 병합 안 됨)
- W&B: 프로젝트 `crane-openvla-oft`, run `ixka58xk`(B, 그룹 `b4-2-lr2e-4`). offline 기록을 별도 wandb 0.31 환경으로 동기화했다.

## 1. 왜 다시 학습했나

원래 b4(LR 5e-4, 유효 batch 8)는 14K 부근에서 입력과 상관없이 마그넷별 다수값만 내는 상태로 무너졌다(validation balanced 0.500).
논문은 batch 32~64에서 LR 5e-4를 쓰는데 우리는 batch 8이라 LR이 상대적으로 크다고 보고 LR을 낮춰 시험했다.

| 실험 | 시작점 | LR | 결과 |
|---|---|---|---|
| A (b4-1 코드) | 원래 b4 10K | 5e-5 | 10K → 12.5K 동안 on/off 판정이 하나도 안 바뀜 (balanced 0.691 고정). 붕괴도, 학습도 없음 |
| A' (b4-2 코드) | A의 12K | 1e-4 | 12K → 15K 동안 그대로 (0.691). 마찬가지 |
| **B (b4-2 코드)** | **처음부터** | **2e-4** | 붕괴 없음. 3K에서 validation 최고(0.740). 아래 결과 |

A의 체크포인트에는 "LR 감소 완료" 기록이 저장돼 있어서, A'는 `--reset_lr_schedule True`로 이어받았다(없으면 LR이 1e-5가 된다).

## 2. b4-1 대비 바뀐 코드

### 학습 기록 (학습 계산에는 영향 없음)
- **train 지표를 GPU 전체 평균으로**: 예전에는 W&B 점 하나가 rank 0의 마지막 스텝(샘플 4개)이라 잡음이 컸다. 이제 샘플 기준 합계를
  모든 GPU에서 all-reduce해서, W&B 점은 10스텝 동안 두 GPU의 모든 샘플(80개) 평균, 로그의 `TRAIN` 줄은 500스텝 평균이다(`samples=` 표시).
- **gradient norm 기록**: 본체(LoRA·FiLM·proprio projector, GPU 간 동일)와 action head(rank 0 값)로 나눠 기록한다
  (`grad_body_mean/max`, `grad_head0_mean/max`, W&B `VLA Train/Grad Norm ...`). gradient clipping은 넣지 않았다(아래 참고).

### action head가 GPU 간에 동기화되지 않는 문제 (원본 OFT 코드)
- `run_forward_pass`가 head를 `action_head.module.predict_action(...)`으로 불러 DDP forward를 건너뛴다. 그래서 head의 gradient는
  GPU 간에 평균되지 않고, GPU마다 자기 샘플로 head를 따로 학습하며 **rank 0의 head만 저장**된다. LoRA·FiLM·proprio projector는 정상 동기화된다.
  torch 2.11에서 CPU 2프로세스(gloo)로 재현해 확인했다. b1~b4와 A·A'·B 모두 해당한다.
- B에서는 **학습 방식은 원본 그대로** 두었다(LR만 바꾼 비교를 위해). 대신:
  - **validation은 rank 0의 head로 채점**한다(`rank0_action_head`: 평가 동안 rank 0 head를 모든 GPU에 빌려주고 끝나면 원래 head로 복원).
    그래서 학습 중 VAL 값 = 저장되는 모델의 성능이다(3K로 재평가해 소수 넷째 자리까지 같음을 확인).
  - **gradient clipping은 빼 두었다.** head gradient가 GPU마다 다르면 clipping 계수도 GPU마다 달라져, 동기화되던 LoRA까지 GPU 간에 갈라진다.

### LR 스케줄
- `--val_plateau_decay`: validation balanced accuracy가 최고값보다 0.005 넘게 오르지 못한 것이 3번 연속이면 LR을 한 번 10배 줄인다.
  5K 이후 validation부터만 센다(초반 정체로 일찍 줄지 않게). validation은 체크포인트 저장 **앞**에 돌아서 저장 상태에 결과가 포함된다.

### 재시작 안전장치
- `train_crane.sh`가 첫 실행 때 `<run>/plan.env`에 설정(LR, 감소 규칙, val·저장 간격, MAX_STEPS, BATCH_SIZE·GRAD_ACCUM·NPROC)을 기록하고,
  같은 RUN_ID를 다시 실행하면 환경변수로 주지 않은 값은 plan.env에서 읽는다.
- 이어받을 때 유효 batch(batch × 누적 × GPU 수)가 저장 당시와 다르면 시작 단계에서 에러를 낸다(training_state에 GPU 수도 저장).
- `--resume_from` 직후 첫 스텝도 저장 대상이다(예: 999에서 멈춘 뒤 이어서 1000의 체크포인트가 빠지지 않게).
- `--reset_lr_schedule True`: 이어받을 때 저장된 LR 감소 기록을 무시하고 `--learning_rate`로 시작한다.

### 평가
- `--eval_split test`(`val_crane.sh`의 `EVAL_SPLIT=test`): test split을 validation과 같은 방식(3프레임 간격 전체, rank 0 head, 같은 지표)으로
  채점한다. 결과는 `<ckpt>/test_metrics_stride3.json`. test로 체크포인트를 고르지 않도록 `--val_only`에서만 허용한다.
- `experiments/robot/crane/upload_eval_to_wandb.py`: 평가 결과를 그 체크포인트를 만든 학습 run의 W&B에 `VLA Test/*`(가로축 `checkpoint_step`)로
  올린다. `val_crane.sh`에 `WANDB_SYNC_ENTITY`를 주면 test 평가 뒤 자동으로 실행된다. 학습 환경의 wandb(0.16.6)는 로그인되어 있지 않아
  로그인된 별도 wandb 환경(`WANDB_PY`, 기본 `~/wandb-cli/bin/python`)으로 올린다. 올린 결과 옆에 `.wandb` 표시 파일을 남겨 중복 업로드를 막는다.

## 3. 학습 설정 (B)

| 항목 | 값 |
|---|---|
| 시작 | 처음부터 (base OpenVLA-7B + LoRA rank 32, FiLM, proprio, 카메라 3대) |
| LR | 2e-4, 7K에서 validation 정체로 2e-5로 자동 감소 |
| 유효 batch | 8. 0~11,354: GPU 2장 × batch 2 × 누적 2 (11,354~11,630 구간에 GPU 1장 / batch 변경 후 다시 2장, 11,630~: GPU 2장 × batch 4 × 누적 1) |
| 규제 | 이미지 augmentation(90% 크롭, 밝기·대비·채도·색조), AdamW 기본 weight decay 0.01. **dropout 없음**(LoRA·LLM·head 모두) |
| validation | 1K마다, validation 10개 에피소드 전체를 3프레임 간격(7,390프레임), rank 0 head |
| 기간 | 2026-11-06 02:18 시작 → 12:26 STOP (14,087스텝). 11K 이후 변화가 없어 15K 전에 멈춤 |

GPU 1장 구간(11,354~11,605)에는 head가 매 스텝 샘플 8개를 모두 봤다(2장일 때는 GPU마다 4개). LR 2e-5 구간이라 영향은 작다고 본다.

## 4. 결과

### validation (7,390프레임, 매번 같은 프레임)

| step | train L1 | val L1 | 정확도 | balanced | LR | grad 본체 평균 / 최대 |
|---|---|---|---|---|---|---|
| 1K | 0.210 | 0.471 | 0.811 | 0.714 | 2e-4 | 0.27 / 0.98 |
| 2K | 0.138 | 0.432 | 0.811 | 0.718 | 2e-4 | 0.20 / 1.06 |
| **3K** | 0.118 | **0.400** | **0.833** | **0.740** | 2e-4 | 0.16 / 0.44 |
| 5K | 0.129 | 0.428 | 0.802 | 0.706 | 2e-4 | 0.24 / 3.46 |
| 7K | 0.089 | 0.433 | 0.803 | 0.714 | 2e-4 → 2e-5 | 0.13 / 0.69 |
| 11K | 0.047 | 0.400 | 0.807 | 0.715 | 2e-5 | 0.09 / 0.60 |
| 14K | 0.041 | 0.404 | 0.805 | 0.711 | 2e-5 | 0.12 / 2.24 |

- 원래 b4(LR 5e-4)의 최고는 5K balanced 0.699였다. B는 붕괴 없이 3K에서 0.740.
- 3K 이후 train L1은 0.12 → 0.04로 계속 내려가는데 validation은 오르지 않는다: **과적합**. 최고점 3K는 데이터를 약 0.2 epoch 본 시점이다
  (train 125,570프레임 / batch 8 ≈ 15.7K스텝 = 1 epoch).
- val L1은 3K와 11K가 같다(0.400). LR을 줄인 뒤 맞히던 프레임의 출력이 ±1에 가까워지며 L1만 내려갔고, on/off 판정은 거의 그대로다.

마그넷별 validation 재현율 (ON / OFF):

| 마그넷 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 정답 ON 프레임 | 779 | 3008 | 4187 | 5941 | 6322 | 6322 | 5370 | 4357 | 2206 | 790 |
| 3K | 0 / 1.00 | 0.61 / 0.92 | 0.95 / 0.31 | 0.93 / 0.67 | 0.93 / 0.92 | 0.94 / 0.91 | 1.00 / 0.66 | 0.88 / 0.81 | 0.90 / 0.56 | 0 / 0.91 |
| 14K | 0 / 0.98 | 0.68 / 0.62 | 0.93 / 0.31 | 0.92 / 0.68 | 0.92 / 0.92 | 0.94 / 0.92 | 1.00 / 0.66 | 0.84 / 0.64 | 0.72 / 0.61 | 0 / 0.93 |

- 마그넷 2를 처음으로 켜기 시작했다(이전 b4는 ON 재현율 0~0.06).
- 마그넷 1·10은 validation에서 한 번도 켜지 않는다. 마그넷 3은 꺼져야 할 때도 켜 두는 경향(OFF 재현율 약 0.31)이 모든 체크포인트에서 같다.

### test (8,230프레임, 10개 에피소드 전체를 3프레임 간격)

| | 정확도 | 다수값 고정 출력일 때 | balanced | L1 |
|---|---|---|---|---|
| validation 3K | 0.833 | 0.748 | 0.740 | 0.400 |
| validation 14K | 0.805 | 0.748 | 0.711 | 0.404 |
| **test 3K** | 0.891 | 0.791 | 0.813 | 0.290 |
| **test 14K** | **0.902** | 0.791 | **0.824** | **0.214** |

| 마그넷 (test, ON / OFF) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 정답 ON 프레임 | 431 | 3536 | 4508 | 7161 | 7161 | 7161 | 6994 | 6468 | 5789 | 857 |
| 3K | 0 / 1.00 | 0.74 / 0.99 | 0.79 / 0.36 | 0.98 / 0.91 | 0.98 / 0.91 | 0.99 / 0.91 | 1.00 / 0.84 | 0.82 / 0.87 | 0.79 / 0.91 | 0.49 / 0.98 |
| 14K | 0 / 0.97 | 0.82 / 0.97 | 0.79 / 0.37 | 0.995 / 0.91 | 0.994 / 0.91 | 0.996 / 0.91 | 1.00 / 0.81 | 0.86 / 0.90 | 0.83 / 0.96 | 0.49 / 0.98 |

- **validation과 test의 순위가 반대다**(validation은 3K, test는 14K가 조금 좋음). 둘 다 에피소드 10개라 3K·14K 차이(balanced 약 0.01~0.03)는
  에피소드 구성에 따라 뒤집히는 크기다. test를 보고 체크포인트를 고르지는 않는다(그러면 test가 validation이 된다).
- 두 체크포인트 모두 test balanced 0.81~0.82로 원래 b4보다 확실히 낫다. 마그넷 10은 test에서 절반 정도 켜고(ON 0.49), 1은 여전히 켜지 않는다.

## 5. 남은 문제와 확인한 사실

- **과적합**: dropout이 없고(이미지 augmentation과 weight decay 0.01뿐), 지시문이 에피소드마다 달라(강판 번호) 에피소드를 외울 수 있다.
- **드문 마그넷**: 1(ON 약 5~11%)은 한 번도 켜지 않는다. L1의 최적해가 다수값이라 드문 ON을 배우기 어렵다.
- **state는 실시간 센서가 아니라 ERP 기록이다.** 데이터 제공 측 문서(`유니스트_v2_포맷_차이.pdf`)에 "U 이전은 0, U부터는 최근 이벤트 행의
  CRNWGT를 홀드", "[X 주행, Y 횡행, Z 권상]을 홀드"라고 되어 있고, 75개 에피소드 모두 state가 ERP 이벤트(U/C/D) 프레임에서만 바뀐다
  (에피소드당 2~30번). U 이전의 0은 측정값이 아니라 "기록 없음"이고, 마그넷 자체 무게 같은 기본 하중은 데이터에 없다.
  U 이후 하중은 D에서 0이 되는 것(66/75)으로 보아 순중량(강판 무게)으로 보인다(추정).
  state가 0에서 값으로 바뀌는 U는 첫 마그넷 ON보다 중간값 6프레임 앞이라, 모델이 영상 대신 이 변화를 보고 켤 수 있다.
  마그넷 세기(`magnet_power_level`)는 다른 출처(DATA csv, 작업자 패널)이고 b4에서는 쓰지 않는다.
- **validation 10개로는 체크포인트 순위가 불안정하다.**

## 6. 명령 (`b4-2/` 안에서, `PATH`에 conda env bin)

```bash
# 학습 (처음부터, LR 2e-4, validation 정체 시 10배 감소). 끊기면 같은 RUN_ID로 다시 실행하면 plan.env와 resume_latest로 이어감
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 BATCH_SIZE=2 GRAD_ACCUM=2 DATA_ROOT=~/datasets/rlds BASE_MODEL=<openvla-7b 사본> \
RUN_ID=crane_b_magnet_onoff_state_tvt--lr2e-4 MAX_STEPS=15000 SAVE_FREQ=1000 VAL_FREQ=1000 VAL_BATCH_SIZE=4 \
LEARNING_RATE=2e-4 VAL_DECAY=1 VAL_DECAY_PATIENCE=3 VAL_DECAY_MIN_DELTA=0.005 VAL_DECAY_MIN_STEP=5000 \
  bash experiments/robot/crane/train_crane.sh

# 저장된 체크포인트 재평가 (validation, 학습과 같은 방식)
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 DATA_ROOT=~/datasets/rlds BASE_MODEL=<같은 base 사본> \
  bash experiments/robot/crane/val_crane.sh checkpoints/crane_b_magnet_onoff_state_tvt--lr2e-4--3000_chkpt

# test 평가 + W&B 업로드 (마지막에만)
EVAL_SPLIT=test WANDB_SYNC_ENTITY=<W&B entity> CUDA_VISIBLE_DEVICES=0,1 NPROC=2 DATA_ROOT=~/datasets/rlds BASE_MODEL=<같은 base 사본> \
  bash experiments/robot/crane/val_crane.sh checkpoints/crane_b_magnet_onoff_state_tvt--lr2e-4--3000_chkpt
```

## 7. 다음 실험 후보 (`b4-3/`)
- state 의존도 확인: 평가 때 state를 0으로 지우거나 섞어서 성능 비교 (학습 없이 가능)
- 드문 마그넷: L1 대신 마그넷별 BCE + `pos_weight`, 또는 ON 가중치
- 과적합: LoRA dropout, 지시문 고정, 학습 길이 단축
- action head gradient 동기화(정상 data-parallel)
- validation 에피소드 늘리기, 서버 평가(`eval_server_on_rlds.py`)로 on/off 전환 시점 오차 비교
