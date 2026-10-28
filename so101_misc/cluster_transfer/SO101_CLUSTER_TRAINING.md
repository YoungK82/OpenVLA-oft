# SO-101 bidirectional 학습: Hawk(SLURM) 실행 가이드

Hawk 로그인: `ssh -p <PORT> <USER>@<HAWK_LOGIN_HOST>` (hlogin01). 작업 폴더는 `/home/<USER>/yhkim/so101-openvla-oft/`이다. 이 계정은 공용이므로 모든 파일을 이 폴더 안에 둔다.

계정 제한(account `n-gpu`, QoS `n-qos`): job당 최대 12시간, 동시에 GPU 최대 8장, `-n`은 GPU당 16코어, 동시 job 15개, 제출 job 20개.

## 목표와 설정

OpenVLA-OFT 논문의 학습 방식과 종료 기준을 따른다. LR 감소 시점과 학습 종료를 고정된 스텝으로 정하지 않고, 학습 L1을 보고 자동으로 정한다.

| 항목 | 논문 ALOHA (scoop, 45 demos) | 이번 학습 |
|---|---|---|
| attention | bidirectional (병렬 디코딩) | bidirectional (`transformers-openvla-oft` fork의 sdpa) |
| 실효 배치 | 32 | 32 (GPU당 2 × GPU 수 × gradient accumulation, A100 40GB). accumulation은 1장이면 16, 2장이면 8로 자동 계산된다 |
| 종료 기준 | 정규화 L1 < 0.01 | **동일.** LR을 낮춘 뒤 L1이 0.01 아래에서 더 이상 줄지 않으면 종료한다. 최종 체크포인트를 저장하고 `DONE` 파일을 만든다 |
| LR | 5e-4 → 5e-5. loss가 매우 느리게 줄어들 때 한 번 감소 | 5e-4. 학습 L1이 2,000스텝 동안 2% 미만으로 줄면 5e-5로 한 번 감소한다(10,000스텝 이후에만, 늦어도 40,000스텝에는 감소) |
| 총 스텝 | 50,000 (scoop) | 위 기준으로 정해진다. 상한은 50,000 |
| LoRA / 이미지 증강 / loss | rank 32 / crop + color / L1 | 동일 |
| 입력 | 3 images + proprio, FiLM | 3 images (front, top, wrist) + 6D proprio, FiLM 없음 (지시문 1개) |
| 체크포인트 | — | 2,500스텝마다 저장 |

기본 요청은 GPU 1장, `-n 16`이다. 사람이 적을 때 `echo 2 > checkpoints/so101-bidir-b32/NEXT_GPUS`를 실행하면, 다음 job부터 GPU 2장(`-n 32`)으로 이어서 학습한다. 실효 배치 32와 optimizer 상태는 GPU 수와 상관없이 그대로 유지된다. job 하나는 최대 12시간이다. 각 job은 11시간 35분이 되면 학습 상태를 저장하고 정상 종료한다. 이어서 다음 job을 스스로 제출하고, 저장된 지점부터 정확히 이어서 학습한다.

이어 학습할 때는 원래 base 모델에 같은 LoRA를 다시 붙이고 optimizer 상태도 복원한다. 예전 bidir 학습에서는 병합된 모델에 새 LoRA를 다시 만들었고, 그 때문에 성능이 떨어졌다. 이번 방식에는 그 문제가 없다.

학습 로그(`train.log`)에서 학습 상태를 바로 확인할 수 있다.
- 500스텝마다: `TRAIN step=... l1_block=... l1_window=... improvement=... lr=...`
- validation마다: `VAL step=... loss=...`
- LR을 낮출 때와 학습이 끝날 때도 한 줄씩 남는다.

`plan.env`의 `MAX_STEPS`와 `DECAY_STEP`은 학습 중에 고칠 수 있다. 고친 값은 다음 job부터 적용된다. 학습 L1이 0.01 아래로 내려가지 않으면 자동으로 끝나지 않고 50,000스텝까지 간다. 상한을 늘리려면 `MAX_STEPS`를 키운다. 논문(LIBERO-Goal)에서도 L1이 더 낮은 체크포인트가 항상 더 좋지는 않았다. 그러니 중간 체크포인트를 실제 로봇에서 비교하고 validation L1도 함께 본다. 자동 판단 없이 고정 일정으로 학습하려면 `AUTO_SCHEDULE=0`으로 둔다.

W&B는 offline으로 기록한다. 여러 job에 걸쳐도 하나의 run으로 이어지며, 로그인 노드에서 `wandb sync`로 올린다.

## 1. 이 서버에서 전송 파일 만들기

```bash
cd /home/yhkim/openvla-oft
bash experiments/robot/so101/pack_for_cluster.sh            # 기본 출력: ~/transfer/so101-cluster-YYYYMMDD
rsync -avP ~/transfer/so101-cluster-YYYYMMDD/ <cluster>:<WORK>/so101-transfer/
```

| 파일 | 내용 | 크기 |
|---|---|---|
| `openvla-oft-code.tar.gz` | 커밋된 코드 (`CODE_COMMIT`에 해시 기록) | 작음 |
| `so101-rlds-1.1.0.tar` | 학습 데이터 RLDS 1.1.0 (trim 버전, train 45개 / val 5개 에피소드) | 약 1.5 GB |
| `openvla-7b-47a0ec7.tar` | base 모델 | 약 15 GB |
| `requirements-cluster.txt` | 이 서버 학습 환경의 패키지 버전 | — |
| `SHA256SUMS` | 무결성 확인용 | — |

## 2. 클러스터 환경 준비 (hlogin01, 1회)

압축 해제는 1단계 다음에 끝났다고 가정한다(`openvla-oft/`, `datasets/`, `models/`). 학습 환경은 계정 공용 conda로 만들되, 위치는 이 폴더 안의 `env/`로 지정한다.

```bash
B=/home/<USER>/yhkim/so101-openvla-oft
export PIP_CACHE_DIR=$B/.pip-cache PYTHONNOUSERSITE=1
conda create -y -p $B/env -c conda-forge --override-channels python=3.10
conda activate $B/env
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1     # PyPI 빌드 = CUDA 12.6
cd $B/openvla-oft && pip install -e . && pip install -r $B/transfer/requirements-cluster.txt
```

- **transformers는 반드시 `moojink/transformers-openvla-oft` fork여야 한다.** 일반 transformers를 쓰면 sdpa가 causal로 동작한다. 예전 25k 체크포인트가 의도와 달리 causal로 학습된 원인이 이것이었다. 그래서 학습 스크립트는 시작할 때 fork인지 검사하고, 아니면 학습을 거부한다.
- **torch 버전:** aga01에서는 torch 2.11(cu130, Blackwell)로 검증했다. Hawk는 2.7.1(cu126)을 쓰므로 아래 3단계 smoke test로 먼저 확인한다.

## 3. 짧은 smoke test (GPU 1장, 약 30분)

```bash
salloc -p a100_40g -A n-gpu --qos=n-qos -N 1 -n 16 --gres=gpu:1 -t 01:00:00
srun --pty bash
cd /home/<USER>/yhkim/so101-openvla-oft/openvla-oft
RUN_ID=smoke MAX_STEPS=60 SAVE_FREQ=60 RESUME_SAVE_FREQ=20 AUTO_RESUBMIT=0 \
  bash experiments/robot/so101/slurm_train_bidir.sbatch
```

로그에서 다음을 확인한다.
- `GPUs=1 ... accumulation=16`이 출력되는지
- `(bidirectional sdpa)`가 출력되는지
- 배치 2에서 메모리 부족 없이 학습이 진행되는지
- `Saved resumable state for Step 20`이 출력되는지

`Step 20` 저장을 확인했으면 다른 터미널에서 `touch checkpoints/smoke/STOP`을 실행한다. 그다음 같은 명령을 다시 실행하고 아래를 확인한다.
- `Resuming from ...`이 출력되는지
- `Restored optimizer after N steps`에서 N이 마지막 저장 스텝 + 1인지
- Step 60에서 학습이 끝나는지

배치 2가 메모리 부족으로 실패하면 `BATCH_SIZE=1`로 바꾼다. accumulation은 자동으로 다시 계산된다.

이어서 LR 자동 감소와 종료 기준을 짧게 확인한다. 블록을 5스텝으로 줄이고 종료 기준을 일부러 느슨하게 둔다. 그러면 20스텝에서 LR이 감소하고 40스텝에서 학습이 끝나며 `DONE` 파일이 생겨야 한다.

```bash
RUN_ID=smoke-sched MAX_STEPS=60 SAVE_FREQ=60 RESUME_SAVE_FREQ=20 AUTO_RESUBMIT=0 STOP_L1=100 \
  EXTRA_ARGS="--loss_block_steps 5 --plateau_window_blocks 2 --min_steps_before_auto_decay 10 --plateau_min_improvement 10" \
  bash experiments/robot/so101/slurm_train_bidir.sbatch
grep -E "^TRAIN|LR decayed|finishing" checkpoints/smoke-sched/train.log; cat checkpoints/smoke-sched/DONE
```

GPU 2장으로 바꾸기 전에는 같은 테스트를 `RUN_ID=smoke2`, `-n 32 --gres=gpu:2`로 한 번 돌린다. 여러 GPU 학습은 아직 실제로 실행해 본 적이 없다. STOP을 만들었을 때 두 GPU가 같은 스텝에서 함께 멈추는지 확인한다.

## 4. 본 학습 제출 (1회, hlogin01에서)

```bash
conda activate /home/<USER>/yhkim/so101-openvla-oft/env     # CONDA_EXE를 job에 넘기기 위해 필요
cd /home/<USER>/yhkim/so101-openvla-oft/openvla-oft
sbatch experiments/robot/so101/slurm_train_bidir.sbatch      # 기본: GPU 1장, -n 16, 12시간
echo 2 > checkpoints/so101-bidir-b32/NEXT_GPUS               # (선택) 다음 job부터 GPU 2장
```

- **진행 확인:** `squeue -u $USER`, `tail -f checkpoints/so101-bidir-b32/train.log`. SLURM 출력은 제출한 폴더의 `slurm-so101-bidir-<jobid>.out`과 `.err`에 남는다.
- **학습 계획:** `checkpoints/so101-bidir-b32/plan.env`에 기록된다. 총 스텝, LR 감소 시점, W&B run id가 들어 있다.
- **W&B:** offline 기록은 `checkpoints/wandb/`에 쌓인다. 로그인 노드에서 `wandb login` 후 `wandb sync checkpoints/wandb/offline-run-*`로 올린다.
- **일부러 중단:** `touch checkpoints/so101-bidir-b32/STOP`. 상태를 저장하고 종료하며, 다음 job을 제출하지 않는다. 다시 이어가려면 같은 `sbatch` 명령을 실행한다.
- **오류로 job이 죽은 경우:** 다음 job을 자동으로 제출하지 않는다. 원인을 고친 뒤 같은 명령으로 다시 제출하면 최근 저장 지점부터 이어진다. 저장은 최대 250스텝 간격이다.
- **compute 노드에서 `sbatch`가 막혀 있는 경우:** `AUTO_RESUBMIT=0`으로 두고, job이 끝날 때마다 같은 명령으로 수동 제출한다.

## 5. 중간 체크포인트를 이 서버로 가져와 추론

체크포인트는 `checkpoints/so101-bidir-b32--{2500,5000,...,25000}_chkpt`에 병합하지 않은 상태로 저장된다. 추론에 필요 없는 optimizer 상태는 빼고 가져온다.

```bash
# 이 서버에서
rsync -avP -e "ssh -p <PORT>" --exclude training_state.pt \
  <USER>@<HAWK_LOGIN_HOST>:/home/<USER>/yhkim/so101-openvla-oft/openvla-oft/checkpoints/so101-bidir-b32--5000_chkpt checkpoints/
conda activate openvla-oft
python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint /home/shared/openvla-oft/models/openvla-7b-47a0ec7 \
  --lora_finetuned_checkpoint_dir checkpoints/so101-bidir-b32--5000_chkpt
bash experiments/robot/so101/serve_portable.sh checkpoints/so101-bidir-b32--5000_chkpt
```

- **병합은 추론할 GPU가 있는 이 서버에서 한다.** OFT 저자들도 테스트할 장치에서 병합하라고 권장한다.
- **attention은 자동으로 맞춰진다.** `training_attention.json`이 체크포인트에 함께 저장되므로, `serve_portable.sh`가 bidirectional sdpa로 서빙한다. 단 클러스터의 transformers 설치 URL이 이 서버와 다르면 서빙을 거부한다.
- **실제 로봇 전 확인:** RLDS validation 재생 오차를 보고, 실제 로봇의 첫 요청에서 `첫 목표 − 현재 state`가 학습 범위(shoulder_lift 약 ±12°) 안인지 본다.
