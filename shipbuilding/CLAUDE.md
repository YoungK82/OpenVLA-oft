# shipbuilding/: 조선소 마그넷 크레인 데이터로 OpenVLA-OFT 학습

이 폴더는 크레인 B 데이터(강판 들기·옮기기·내려놓기)로 OpenVLA-OFT를 학습해 **마그넷 10개의 on/off를 예측**하는 작업이다.
SO-101(`../so101/`)을 새 하드웨어에 붙였던 방식을 그대로 따른다. 사용자와는 한국어로 대화한다.

## 레포 규칙
- 레포 `YoungK82/OpenVLA-oft`(public). 루트는 원본 `moojink/openvla-oft`와 **동일하게 유지**하고 수정하지 않는다.
- `so101/` = SO-101 bidirectional 학습 파이프라인, `so101_misc/` = 잡동사니, `shipbuilding/` = 이 작업.
- 각 하위 폴더는 독립 사본(자체 `prismatic/`, `vla-scripts/`)이다. 실행은 그 폴더 안에서 `PYTHONPATH=$PWD`로 한다.
  conda 환경 `openvla-oft`의 editable 설치는 `../so101/`을 가리키므로 PYTHONPATH를 꼭 지정한다.
- transformers는 `moojink/transformers-openvla-oft` fork여야 bidirectional로 학습된다(이 서버 env는 이미 fork).
- 데이터·RLDS·체크포인트·manifest는 커밋하지 않는다(public). push는 사용자가 토큰으로 직접 한다.
- md 파일을 늘리지 않는다. 사용법은 스크립트 상단 주석에 둔다.

## 데이터 (`/home/shared/2026shipbuildingAX/Data_201012-201014/`)
- 75개 폴더(이름=TID), 폴더마다 LeRobot v3 데이터셋 1개(에피소드 1개). 30fps, 1920×1080, 16~221초, 총 180,105프레임.
- 날짜별 13(10/12) / 41(10/13) / 21(10/14)개. 데이터 제공 측 설명: 폴더 안 `유니스트_v2_포맷_차이.pdf`.
- 학습용 parquet `data/chunk-000/file-000.parquet`:
  - `action` (10): magnet_1~10 on/off, 0/1 (크레인 PLC 여자 명령). 에피소드당 바뀌는 횟수 중간값 2.
  - `observation.state` (4) = `observation.load`(1) + `observation.crane_pos`(3): `[load_total_t, x_주행, y_횡행, z_권상]`.
    센서 실측이 아니라 **ERP 이벤트 값을 유지**한 것. U 이벤트(150번째 프레임, 5초)까지 전부 0, 이후 6~13초마다 바뀜.
  - `observation.magnet_power_level` (1): 작업자 패널의 자력 단계. 5단 76.5%, 3단 4.1%, 0단 0.3%, 4단 3프레임, NaN 19.1%
    (NaN = 단수 꺼짐). **숫자면 99.9% 마그넷 ON, NaN이면 28% ON → on/off 정답이 거의 새는 값.**
  - `current` (10): 이번 배치는 전부 NaN → 사용 안 함.
- `ship_ax_excite_measured.parquet`(측정여자)는 학습 데이터가 아니다(명령보다 늦게 따라오는 결과값). 쓰지 않는다.
- `meta/episodes/*.parquet`: `lift/lower_event_frame`, `erp_events`, `plate_id` 등 → 입력 금지(정답 시점), 평가 구간 나누는 데만 사용.
- 영상: parquet 행 n = 각 mp4의 n번째 프레임(0초 시작, CFR). 카메라 9대 중 exo 3대만 사용.

## 결정된 설계
- 교수님 방안: 1) parquet 정보를 언어로 넣어 모델 변경 없이 학습(먼저 할 것) → 1.5) 강재 스펙은 언어, 로드셀은 projector →
  2) projector 여러 개. 사용자 해석으로 1번은 **state = OFT proprio 토큰(ALOHA qpos와 같은 방식), 나머지 parquet 값 = 언어**.
- 두 변형(차이는 `experiments/robot/crane/constants.py`와 학습 스크립트의 데이터셋 이름뿐):
  - `b1_with_power/`: 지시문 = task + `. 마그넷 세기 N단`(NaN → `꺼짐`), 데이터셋 `crane_b_magnet_onoff_power`
  - `b2_without_power/`: 지시문 = task만, 데이터셋 `crane_b_magnet_onoff_nopower`
  - 둘 다 state 4차원을 proprio로 넣는다(`--use_proprio True`). b1 vs b2로 마그넷 세기의 효과(누수)를 본다.
  - 이전 A안(task만, proprio 없음)과 "하중·위치까지 텍스트로" 안은 폐기했다.
- 카메라: cam1 = exo_1(CAM1), cam2 = exo(CAM2), cam3 = exo_3(CAM3) → RLDS `cam1/2/3_image` → `primary/secondary/tertiary`.
- 256×256으로 저장(1920×1080을 눌러서 축소), 모델 입력 224.
- `NUM_ACTIONS_CHUNK = 8`(약 0.27초), `ACTION_DIM = 10`, `PROPRIO_DIM = 4`, 정규화 `BOUNDS`(0/1 → -1/1, 그대로 복원).
- FiLM 사용(`--use_film True`): 지시문이 에피소드마다 다르므로. FiLM은 지시문 토큰 임베딩 평균을 쓰므로 숫자는 거의 전달 안 됨.
- validation 10개 = 날짜별 무작위 1/6/3개(seed 7). `constants.py`의 `VALIDATION_PER_DATE`.

## state가 모델에 들어가는 경로 (원본 OFT 코드 그대로, 새 구현 없음)
`state` → `observation/proprio` → 데이터셋 통계 min/max로 차원별 [-1, 1] 정규화(`prismatic/vla/datasets/rlds/utils/data_utils.py`,
통계는 체크포인트 `dataset_statistics.json`에 저장) → `ProprioProjector`(Linear 4→4096, GELU, Linear; `prismatic/models/projectors.py`)
→ 영상 패치 토큰 뒤에 토큰 1개로 붙음(`prismatic/extern/hf/modeling_prismatic.py` `_process_proprio_features`).
추론 때는 `experiments/robot/openvla_utils.py` `normalize_proprio`가 체크포인트 통계로 같은 정규화를 한다.

## so101/ 대비 바꾼 코드 (두 폴더 동일)
- `prismatic/vla/constants.py`: `CRANE_CONSTANTS`, `--robot_platform crane`, 이름 자동 인식(`crane`).
- `oxe/configs.py`(인코딩 `CRANE_LOAD_POS`, `MAGNET_ONOFF_10` + 항목 `crane_b_magnet_onoff`), `oxe/transforms.py`, `oxe/mixtures.py`.
- `oxe/materialize.py`: 인코딩 허용·마스크, 등록 안 된 `crane_*` 이름은 공용 항목 사용(`registry_name`).
- `prismatic/vla/datasets/datasets.py`: 이름에 `crane`이 있으면 카메라 3대 로드, 이미지 순서에 `image_tertiary`.
  **이게 빠지면 에러 없이 카메라 1대만 학습된다.**
- `vla-scripts/finetune.py`(CRANE: 이름에 crane, 카메라 3대, proprio 필수), `vla-scripts/deploy.py`(카메라 3대).
- `experiments/robot/crane/`: `constants.py`, `crane_utils.py`(`instruction_text` — 변환과 추론이 같이 씀),
  `convert_crane_to_rlds.py`, `verify_rlds_dataset.py`, `train_crane.sh`, `serve_crane.sh`, `eval_server_on_rlds.py`.

## 명령 (각 폴더 안에서, `PY=~/miniconda3/envs/openvla-oft/bin/python`)
```bash
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES= $PY experiments/robot/crane/convert_crane_to_rlds.py \
  --source-root /home/shared/2026shipbuildingAX/Data_201012-201014 --output-root ~/datasets/rlds   # CPU, 약 30프레임/s
$PY experiments/robot/crane/verify_rlds_dataset.py --data-root ~/datasets/rlds
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/datasets/rlds BASE_MODEL=<openvla-7b> bash experiments/robot/crane/train_crane.sh
#   병합: vla-scripts/merge_lora_weights_and_save.py (학습은 --merge_lora_during_training False 기본)
CUDA_VISIBLE_DEVICES=0 VLA_PORT=8777 bash experiments/robot/crane/serve_crane.sh <checkpoint>
$PY experiments/robot/crane/eval_server_on_rlds.py --data-root ~/datasets/rlds --server-endpoint http://127.0.0.1:8777/act
```
- base 모델 `/home/shared/openvla-oft/models/openvla-7b-47a0ec7`은 공유 폴더다. `finetune.py`가 `--vla_path`의 `config.json`을
  고쳐 쓰므로, 테스트 때는 safetensors만 링크하고 나머지는 복사한 사본 폴더를 `BASE_MODEL`로 쓴다.
- 평가 지표: 마그넷별 정확도(0.5 기준), 10개 모두 맞춘 비율, on/off 바뀌는 시점 오차(프레임)와 놓친 개수.

## 학습이 어떻게 돌아가는가 (finetune.py, 모델 코드는 원본 그대로)
- 한 샘플 = cam1~3 이미지(각 224, 패치 256개 → 768토큰) + 지시문(약 90토큰) + state proprio 토큰 1개 + action 자리 80토큰(8×10).
  LLM(Llama-2 7B)이 이 약 940토큰을 bidirectional로 한 번에 보고, action 자리 80개의 출력을 L1 회귀 헤드가 (8, 10) 값으로 바꾼다.
  정답은 0/1을 -1/1로 정규화한 값, 손실은 L1. 추론 때는 되돌린 뒤 0.5 기준으로 on/off.
- 학습 대상: LoRA(rank 32, LLM과 비전 인코더의 모든 linear) + FiLM 층 + proprio projector + action head. 7B 본체는 고정.
- 옵티마이저 AdamW(lr 5e-4, 기본값), 스케줄 MultiStepLR 10배 감소 1회. `--auto_lr_decay`면 train L1이 정체될 때 감소
  (`MIN_DECAY_STEP` 이후, 늦어도 `DECAY_STEP`). OFT+ ALOHA 원본은 8 GPU×batch 4, 50K에서 감소, 100K steps.
- 데이터 크기: train 65 에피소드 157,865프레임, validation 10 에피소드 22,240프레임. batch 8이면 1 epoch ≈ 19.7K steps.
- **메모리: batch 8은 약 83GB 이상 필요** (gradient checkpointing 없음, 샘플당 약 940토큰의 활성값 + 비전 인코더 역전파).
  aga01 GPU 1(96GB, 다른 사용자 13GB 사용 중)에서 batch 8은 OOM(81.8GB에서 실패, 2026-10-29). SO-101도 batch 8에 약 88GB가 필요했다.
  → aga01은 `BATCH_SIZE=4 GRAD_ACCUM=2`(유효 8) 권장. Hawk A100 40GB: batch 2는 첫 스텝 OOM(39.5GB 꽉 참), batch 1은 정상
  (2026-10-29 측정). GPU 2장 × batch 1 × 누적 4 = 유효 8에서 약 3.2초/스텝 → 25K ≈ 22시간.
- 과적합 주의: 지시문 75개가 전부 다름(강판 번호 = 사실상 에피소드 ID)이라 FiLM으로 에피소드별 타이밍을 외울 수 있다.
  체크포인트는 train L1이 아니라 validation(마그넷별 정확도, 10개 모두 맞춘 비율, on/off 시점 오차)으로 고른다.
- 같은 GPU 종류에서 병합·평가한다(OFT 권고). 체크포인트는 병합 안 된 상태로 저장된다(`--merge_lora_during_training False`).

## 학습 전에 사용자가 정할 데이터 특성 (교차 확인에서 확인됨, 버그 아님)
첫 Hawk 학습(2026-10-29)은 아래를 모두 **그대로 둔 채**(패딩 유지, 1프레임 에피소드 포함) 진행했다. 바꾸려면 aga01에서 재변환.
1. 모든 에피소드 앞뒤 150프레임(5초)씩 패딩: state 전부 0, 마그넷 세기 NaN, 마그넷 전부 OFF. 전체의 12.5%.
   첫 ON이 75개 중 68개에서 정확히 156번째 프레임 → "state 0이면 OFF", "5초 뒤 켜기" 같은 지름길 위험.
   정규화 최솟값도 0이 되어 실제 x는 약 [-0.73, 1], y는 약 [-0.68, 1]만 씀. 유지할지 잘라낼지(자르면 재변환) 미정.
2. 마그넷이 1프레임만 켜진 train 에피소드 2개(`2026101315232601`, `2026101416333501`): 제외할지 미정.
3. validation에만 있는 패턴(z만 0인 프레임, 조합 `0000111100` 등) → val 정확도가 모델과 무관하게 낮을 수 있음.
4. b1: 마그넷 세기 문구가 정답을 거의 드러냄(의도된 설계, b2와 비교). `0단`(10개 에피소드)과 `꺼짐`(NaN)이 다른 문구 — 같게 볼지 미정.
5. on/off가 바뀌는 순간은 action chunk의 0.8%뿐 → 평균 정확도보다 시점 오차로 평가.

## 서버·GPU 규칙
- 공용 서버(aga01, RTX PRO 6000 96GB × 2). **GPU를 쓰기 전에 어느 GPU를 쓸지 사용자에게 알리고, 다른 사용자의 사용량을 먼저 확인한다.**
  (2026-10-29: GPU 0은 다른 사용자가 92GB 사용 중이어서 사용자가 GPU 1로 지정했다.)
- 오래 걸리는 작업(변환, 학습)은 tmux에서 돌린다.
- 작업(백그라운드 작업, tmux 실행, 테스트, 변환, 파일 작성 등)이 하나 끝나면 **그 즉시 결과를 보고한다.**
  다른 작업이 끝나기를 기다렸다가 한꺼번에 보고하지 않는다. 여러 작업이 동시에 돌면 끝난 순서대로 짧게 보고하고,
  아직 돌고 있는 작업을 함께 적는다. 대기 중이면 무엇을 기다리는지, 대략 언제 끝나는지 알린다.
- Hawk 클러스터(SLURM, job 12시간 제한, A100 40GB)용 SO-101 스크립트는 `../so101_misc/cluster_transfer/`에 있다(자동 재제출 참고용).
- 크레인 Hawk 학습: 각 폴더 `experiments/robot/crane/slurm_train_crane.sbatch`(사용법은 상단 주석, 두 폴더는 폴더·데이터셋·job 이름만 다름).
  - 이 계정은 SO-101 문서의 `n-gpu`/`n-qos`(12시간)가 아니라 **account `m-gpu` / QoS `m-qos`**(job당 최대 36시간, GPU 최대 4장).
  - job 24시간 = job 하나의 시간일 뿐, 학습 길이가 아님. 23시간 35분에 STOP → 저장 → 자기 자신 재제출 → `--resume_from`으로
    정확히 이어감(250스텝마다 resume 상태 갱신). MAX_STEPS 도달 또는 수동 `touch <run_dir>/STOP`이면 끝. 오류 종료는 연속 2번까지만
    재제출(`CRASH_COUNT`), 노드 장애는 `--requeue`. GPU 2장 STOP·resume은 Hawk smoke로 확인함(2026-10-29).
  - 기본값: GPU 2장, 유효 batch 8(1×2×누적 4), **25K 스텝(≈1.3 epoch), LR 5e-4 고정(감소 없음)** — 사용자 결정. `LR_DECAY=1`이면
    10K~20K에 자동 감소. 예전 120K(train_crane.sh)는 논문 스텝 수를 batch 차이 없이 가져온 값이라 버림(논문 ALOHA는 batch 32×100K,
    LIBERO는 batch 64×150K). 25K에서도 validation이 좋아지면 `<run_dir>/plan.env`의 `MAX_STEPS`만 늘려 이어서 학습.
  - job 이름: `yhkim-crane-oft-power`(b1), `yhkim-crane-oft-nopower`(b2). 출력은 제출 폴더의 `slurm-<job이름>-<jobid>.out`.
  - Hawk 경로: env·base 모델은 `~/yhkim/so101-openvla-oft/`(env는 transformers fork), RLDS는 `~/yhkim/datasets/rlds/`.
    스크립트는 `conda activate` 앞뒤로 `set +u`/`set -u` 필요(conda hook이 정의 안 된 변수를 읽어 job이 바로 죽음).
- 접속 정보는 레포에 적지 않는다. Hawk 계정은 공용이므로 비밀번호·토큰을 파일이나 프롬프트에 남기지 않는다.
- Hawk에서는 Claude를 로그인 노드에서 실행하고(계산 노드는 인터넷 없음), 학습은 sbatch로 돌린다.
  코드는 한 곳에서만 고치고 GitHub로 맞춘다(두 세션이 같은 파일을 동시에 고치지 않는다).

## 진행 상황 (2026-10-29 10:50)
- 완료: b1·b2 구현, 75개 전체 RLDS 변환(aga01 `~/datasets/rlds/crane_b_magnet_onoff_{power,nopower}/1.0.0`, 각 20GB,
  train 65 / val 10), 전체 검증 통과, 교차 확인(모델 코드 원본과 동일, parquet·영상·카메라·프레임 일치, 학습 배치 정상) 통과.
- `finetune.py`에 exact resume·자동 LR 감소 이식, STOP→재시작 테스트 통과. smoke(10스텝, FiLM+proprio, 병합→서버 (8,10)) b1·b2 통과.
- 커밋 `abc89c0`(shipbuilding 추가). push는 사용자가 토큰으로 직접 한다(이 서버엔 GitHub·Hawk SSH 키가 없음).
- RLDS power·nopower 둘 다 Hawk `~/yhkim/datasets/rlds/`로 전송 완료(각 75개 파일, 20GB).
- Hawk SLURM 스크립트 추가, smoke(OOM 확인 → batch 1, STOP·resume Step 23→60) 통과.
- **Hawk에서 학습 중 (2026-10-29 10:33~, hawk02):** b1 power = job 56520, b2 nopower = job 56522. 각 GPU 2장, 25K 스텝,
  약 22시간 예상. 체크포인트 `<폴더>/checkpoints/crane_b_magnet_onoff_{power,nopower}--{5000,...,25000}_chkpt`(병합 안 됨).
- 다음: 체크포인트를 aga01로 가져와(`training_state.pt` 제외) 병합 → `serve_crane.sh` → `eval_server_on_rlds.py`로 b1·b2 비교.
- Hawk에서 추가한 `slurm_train_crane.sbatch` 2개와 이 파일 수정은 아직 커밋 안 함.
