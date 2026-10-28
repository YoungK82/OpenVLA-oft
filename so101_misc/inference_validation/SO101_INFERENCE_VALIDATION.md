**최근 파이프라인 점검 — 2026-10-12**

현재 서버(최신 변경): 사용자 요청으로 기존 `so101-b8-s25000-trimmed-job-1--25000_chkpt`를 causal `eager` 설정으로 로드했다. 주소는 `127.0.0.1:8777`이며 로그는 `outputs/causal-25000-20261012/`이다. 저장 샘플 HTTP 200, 유한한 `(30, 6)` 응답 확인. 클라이언트 프로토콜은 bidirectional 모델과 동일하다. 이 설정은 causal 의미를 유지하되, 과거 서버의 attention 커널/환경을 완전히 재현한 것은 아니다. 아래 7500/15000 내용은 이전 검사 기록이다.

학습 데이터 추가 감사: 실제 RLDS train 45개 episode/27,387프레임, validation 5개/2,264프레임을 검사했다. train의 `action[t] - state[t]`에서 shoulder_lift 양의 최대값은 +12.40°, elbow_flex 음의 최소값은 -16.44°였다. 최신 dry run 첫 예측의 +54.21°/-38.13°는 각각 이를 벗어난다. 원본 trimmed Parquet와 RLDS의 상태·행동 전체 연속열은 50/50 episode에서 정확히 일치해 변환 중 추가 시간 밀림은 발견되지 않았다. 세부 결과: [학습 데이터 감사](outputs/training-action-audit-20261012/REPORT.md). 원본 촬영 시점 동기화 및 실제 영상과의 일치는 별도 미검증이며 causal 비교는 수행하지 않았다.

모델 구성 추가 감사: 7500 학습 로그는 `Attention implementation: sdpa`와 Step 7500 병합 저장을 기록한다. 학습 메타데이터 및 서버 시작 설정은 OFT fork SDPA로 일치한다. 설치된 Llama SDPA는 causal mask를 양방향 mask로 변환하고 `is_causal=False`를 사용한다. 같은 구현의 소형 CPU 모델에서도 마지막 토큰 변경이 첫 토큰 출력에 영향을 주는 것을 확인했다(`outputs/bidir-7500-20261012/attention-implementation-check.json`). 실행 중 GPU 객체를 직접 덤프한 검사는 아니다.

7500 디렉터리에는 같은 스텝 action head와 proprio projector가 있고, 로더는 이 디렉터리에서 strict state-dict 로딩한다. 입력 3장, proprio 6개, L1 head, 출력 30×6은 학습/배포 설정과 일치한다. 기존 causal 계열 25k와 dataset_statistics 및 preprocessor 설정 파일의 SHA-256이 같다. 현재 구조/설정에서 causal로 잘못 실행되거나 다른 스텝 head를 선택한 증거는 발견하지 못했다.

단, 이 모델은 기존 25k를 attention만 바꿔 실행한 모델이 아니다. 별도 bidirectional 학습 계열이며, 7500은 5000 병합 체크포인트에서 새 LoRA를 붙여 학습을 이어간 결과다. 재개 시 optimizer/scheduler 상태는 복원되지 않았다. 따라서 기존 causal의 실기동 성공과 현재 bidirectional 실패만으로 attention 자체를 원인으로 분리할 수 없다. 실제 입력을 고정한 모델 비교와 학습/실행 입력 대조가 아직 필요하다.

최신 교체: 사용자 요청으로 서버를 `so101-bidir-b8-s10000--7500_chkpt`로 변경했다. OFT fork의 bidirectional SDPA, `127.0.0.1:8777` 유지. 저장 샘플 요청에 유한한 `(30, 6)` 응답을 확인했으며 실제 로봇 명령은 보내지 않았다. 로그는 `outputs/bidir-7500-20261012/`에 저장한다. 아래 파이프라인 검사는 교체 전 15k 기준이며, 2026-09-28의 서버 종료/25k 설명도 당시 검사 기록이다.

- 저장된 샘플을 현재 `/act`에 다시 전송: HTTP 성공, 유한한 `(30, 6)` 응답, 서버 진단 로그의 action과 응답 일치.
- 업로드된 Windows 코드의 카메라 계약은 `front → top → wrist`. 서버도 같은 순서로 처리한다. 실제 카메라 장면, RGB 색상, 프레임 신선도는 저장된 서버 로그에 없어 이번 검사로 확인하지 못했다.
- 업로드된 Windows `_run_episode` 함수를 모의 로봇과 기록된 청크로 90스텝 실행: 요청 시점 `0, 30, 60`, 모의 `send_action` 90회. 청크 순서, 관절 이름 매핑, 현재 상태 대비 ±5 제한 결과가 모두 일치했다. 실제 로봇에는 명령을 보내지 않았다.
- 클라이언트는 동기 추론이므로 청크 사이에 대기한다. 기존 실제 요청 간격 중앙값은 약 1.31초이며, 30Hz 연속 실행을 보장하는 구조가 아니다. 제한을 적용한 목표는 원래 예측과 달라질 수 있다.
- `send_action` 반환값과 매 스텝 실제 관절 위치가 기존 클라이언트 로그에 없으므로, 현재 Windows에서 최종 전송된 값과 모터 추종까지 완전 검증한 것은 아니다. 이번 모의 실행은 업로드된 코드 기준이며 현재 Windows 파일 동일성도 별도 확인이 필요하다.

검사 결과: `outputs/bidir-15000-20261012/pipeline-check.json`. 통신·청크 처리 점검 통과는 작업 성공이나 바닥 접촉이 없는 궤적을 보장하지 않는다.

---

**SO-101 25k 실제 추론 검증 결과와 Windows 연결 방법 — 2026-09-28**

현재 서버에서 25k 체크포인트의 전체 GPU 로딩과 HTTP 추론을 검증했다. 기록 데이터의 정상 요청 25건은 모두 통과했다. 잘못된 state를 받아들이던 서버 입력 검증을 보완했고, 실제 요청으로 재검증했다. 로봇 연결이나 모터 명령은 실행하지 않았다.

검사 서버는 작업 종료 시 정상 종료했다. 현재 8777 listener와 검사 모델의 GPU 점유는 해제된 상태다. 나중에 아래 순서로 다시 실행하면 된다.

**확인한 결과**

| 항목 | 결과 |
| --- | --- |
| 모델 | `checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt`, 병합 모델 + 같은 step의 head/projector |
| 환경 | Conda `openvla-oft`, Python 3.10.21, PyTorch 2.11.0+cu130, OFT Transformers 4.40.1 |
| GPU | RTX PRO 6000 Blackwell GPU 0; 추론 후 프로세스 점유 16,720 MiB, 약 16.3 GiB. peak memory를 별도로 계측한 수치는 아님 |
| 입력 | RLDS 1.1.0 validation 5개 episode에서 각각 5개 frame, 총 25건 |
| 정상 출력 | 25/25 요청에 유한한 `(30, 6)` action. 총 750개 6D 목표 |
| 최종 로컬 HTTP 왕복시간 | 중앙값 0.217초, p95 0.249초, 최소 0.200초, 최대 0.990초. 최대값은 재기동 후 첫 평가 요청 포함 |
| 초기 요청 | 최초 서버 실행의 첫 smoke 요청에서 모델 처리시간 1.477초 |
| 반복성 | 같은 관측 3회 추가 요청의 출력 차이 0 |
| 수정 전후 출력 | 동일한 정상 요청 25건의 전체 750개 action vector가 정확히 일치 |
| 잘못된 입력 | 9/9 HTTP 422로 거부; 오류 입력 이후 정상 요청 복구도 통과 |
| 통신 호환 | 일반 NumPy JSON 및 `encoded` 요청 모두 실제 HTTP 추론 통과. 깨진 JSON은 422, `/openapi.json`은 200 |
| 회귀 검사 | CPU unittest 5개 통과: 입력 거부, 정상 client 형식 유지, 비정상 모델 출력 차단, 내부 오류의 500 응답 |

위 시간은 **같은 Linux 호스트의 loopback 측정값**이다. GPU 두 장 모두 다른 작업이 실행 중인 환경에서 측정했으며, Windows 네트워크·카메라·로봇 처리시간은 포함하지 않는다. 현재 제어 코드는 동기 요청이므로 query 시 기다리는 시간이 생긴다. 이 결과를 30 Hz 모델 추론으로 해석하면 안 된다.

**발견하고 보완한 서버 문제**

기존 서버는 길이 1짜리 state를 NumPy broadcasting으로 처리했고, Inf state를 정규화 과정에서 clip해 정상 action처럼 반환했다. NaN state에서는 NaN action이 HTTP 200으로 반환됐다. 누락된 필드나 잘못된 이미지에도 HTTP 200 + `"error"`를 돌려주었다.

[deploy.py](/home/yhkim/openvla-oft/vla-scripts/deploy.py)에 다음 검사를 추가했다.

- 요청 전에 state의 정확한 차원·숫자 타입·유한값, instruction, 이미지 수·형태·dtype 검사. 실패하면 422.
- 예측 후 정확한 action chunk shape와 유한값 검사. 비정상 출력과 내부 오류는 500으로 응답하고 action을 반환하지 않음.
- 기존 정상 요청 및 이중 인코딩 응답 형식 유지.

9개 실측 오류 입력은 instruction 누락, state 누락, 짧은 state, NaN state, Inf state, 카메라 누락, float 이미지, 잘못된 채널 수, 빈 instruction이다. 출력 NaN/Inf·부분 chunk·내부 예외 차단은 모델 결과를 주입하는 CPU 회귀 검사로 확인했다.

**서버에서 평가할 수 있었던 동작 위험**

| 관절 | 첫 action MAE | 30-step chunk MAE | 첫 목표와 현재 state 차이 최대 | 차이 5 초과 요청 수 |
| --- | ---: | ---: | ---: | ---: |
| shoulder_pan | 0.928 | 2.229 | 2.032 | 0/25 |
| shoulder_lift | 5.434 | 12.087 | 10.808 | 16/25 |
| elbow_flex | 4.108 | 9.591 | 10.534 | 5/25 |
| wrist_flex | 4.668 | 6.771 | 8.876 | 11/25 |
| wrist_roll | 1.341 | 1.489 | 3.111 | 0/25 |
| gripper | 3.602 | 4.613 | 5.061 | 1/25 |

MAE는 기록된 정답 action과의 절대오차 평균이다. 팔 다섯 관절은 기존 데이터의 degree 단위이고 gripper는 별도 기록 단위다. 성공률을 측정한 표가 아니다.

- **25건 중 21건**에서 최소 한 관절의 첫 목표 변화가 기존 제한값 5를 초과했다. 저장소의 `validate_and_limit_action(..., 5)`을 기록 state에 적용한 25건은 모두 변화량 제한을 만족했다. 이는 제한 함수 검사이며 실제 Windows 코드나 움직이는 로봇에서의 검사 결과가 아니다.
- shoulder_lift의 **107/750개 목표**가 학습 action 최소값보다 작았다. 학습 최소값은 약 -99.077, 예측 최소값은 약 -101.965로 최대 초과량은 **2.888도**다. 다른 관절은 샘플 범위에서 학습 min/max 안에 있었다.
- 검사 입력 25건의 state는 모두 학습 min/max 안에 있었다. 학습 범위 밖의 실제 자세에 대한 동작은 검증되지 않았다.
- 학습 min/max는 기계적 관절 한계가 아니다. 서버에 이 범위를 강제로 적용하면 정책 동작을 바꾸므로 이번에는 수치를 기록하고 기존 제어 제한을 유지했다.

판정: **현재 환경에서 추론과 기본적인 잘못된 입출력 차단은 확인됐다. 로봇에 raw action을 바로 보내도 안전하다는 판정은 아니다.** 충돌·토크·모터 과부하·실제 calibration·카메라 순서·실시간 상태 지연은 이번 평가 범위 밖이다. 기존 Windows 클라이언트의 제한 기능을 유지하고 실제 로봇에서 별도로 확인해야 한다.

**나중에 실행할 순서**

서버의 현재 내부 IP는 `10.0.12.117`, SSH port는 **50123**이다. 아래 IP는 같은 LAN/VPN에서 접근할 때의 기준이다. 외부 접속을 사용한다면 현재 사용 중인 SSH 주소·포트로 바꾼다. Windows에서의 실제 접속은 이번 Linux 검사에서 실행하지 않았다.

1. Windows PowerShell A에서 서버에 접속한다.

```powershell
ssh -p 50123 yhkim@10.0.12.117
```

접속한 Linux 셸에서 다음을 실행하고 창을 유지한다. GPU 0은 이번 검사에서 사용한 장치다. 다른 GPU를 사용할 때는 `CUDA_VISIBLE_DEVICES`를 바꾼다.

```bash
source /home/yhkim/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/yhkim/openvla-oft
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export VLA_HOST=127.0.0.1
export VLA_PORT=8777
mkdir -p outputs/so101-runtime
export VLA_DIAGNOSTIC_LOG="$PWD/outputs/so101-runtime/actions.jsonl"
bash experiments/robot/so101/serve_portable.sh \
  "$PWD/checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt"
```

Uvicorn이 `http://127.0.0.1:8777`에서 준비됐다는 로그까지 기다린다. 설치나 모델 다운로드는 이번 검증에서 필요하지 않았다. 검사는 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`에서도 성공했다.

2. Windows PowerShell B를 새로 열어 SSH 터널을 유지한다.

```powershell
ssh -p 50123 -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8777:127.0.0.1:8777 yhkim@10.0.12.117
```

이 창이 출력 없이 유지되는 것이 정상이다. Windows의 `127.0.0.1:8777`이 Linux 추론 서버로 전달된다. 포트 충돌이 나면 터널 왼쪽 8777을 18777 등으로 바꾸고 아래 Windows URL·client port도 함께 바꾼다.

3. Windows PowerShell C에서 **로봇 없이** 실제 추론까지 확인한다.

```powershell
curl.exe --fail http://127.0.0.1:8777/openapi.json
scp -P 50123 yhkim@10.0.12.117:/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/validated/sample-request.json .\so101-sample-request.json
curl.exe --fail --max-time 120 -H "Content-Type: application/json" --data-binary "@so101-sample-request.json" http://127.0.0.1:8777/act -o so101-sample-response.json
```

이 파일은 이번 검사에서 통과한 기록 관측이다. 기존 Windows client 환경을 활성화한 뒤 다음으로 응답을 확인한다. `json_numpy`와 NumPy는 기존 client 의존성이다.

```powershell
python -c "import json_numpy, numpy as np; from pathlib import Path; a=np.asarray(json_numpy.loads(Path('so101-sample-response.json').read_text())); print(a.shape, np.isfinite(a).all()); assert a.shape == (30,6) and np.isfinite(a).all()"
```

예상 출력은 `(30, 6) True`다. PowerShell의 `curl` 별칭 대신 위와 같이 `curl.exe`를 사용한다. `ssh`의 port는 소문자 `-p`, `scp`의 port는 대문자 `-P`다.

4. 실제 로봇을 사용할 때는 기존 Windows 파일·환경·COM port·robot id·calibration·카메라 설정을 그대로 사용하고, 기존 실행 명령의 서버 옵션만 다음으로 맞춘다.

```textd
--vla_server_url 127.0.0.1 --vla_server_port 8777
```

완전한 URL을 사용하는 기존 명령이라면 `http://127.0.0.1:8777/act`로 지정한다. 처음에는 기존 client의 `--dry_run True`로 짧게 확인한다. 이 client의 dry-run은 로봇 연결·관측을 수행하므로, 로봇이 없는 지금은 3번의 기록 관측 요청을 사용한다. 이전 Windows client는 query당 30개 action을 허용하는 별도 변경이 있었다는 기록이 있으므로 이번 저장소 파일로 덮어쓰지 않는다.

마칠 때 PowerShell A의 Linux 서버에서 Ctrl+C, PowerShell B의 터널에서도 Ctrl+C를 누른다.

**결과 파일과 재검사**

- [최종 평가 JSON](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/validated/report.json), [수정 전 평가 JSON](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/baseline/report.json)
- [최종 서버 로그](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/server-validated.log), [환경 기록](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/environment.json)
- [전체 정상 출력 비교](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/normal-output-comparison.json), [HTTP 호환 검사](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/http-compatibility.json), [CPU 검사 로그](/home/yhkim/openvla-oft/outputs/so101-server-validation-20260928-1407/unit-tests.log)
- [재사용 가능한 평가 도구](/home/yhkim/openvla-oft/experiments/robot/so101/evaluate_server_safety.py), [회귀 검사](/home/yhkim/openvla-oft/experiments/robot/so101/test_server_validation.py)

서버가 실행 중일 때 Linux의 같은 Conda 환경·저장소 루트에서 평가를 다시 실행할 수 있다.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="" TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 \
  python experiments/robot/so101/evaluate_server_safety.py \
  --builder-dir /home/yhkim/datasets/rlds/so101_block_into_cup_50_v5/1.1.0 \
  --checkpoint checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt \
  --output-dir outputs/so101-safety-recheck
```

모델 로더가 config 백업 파일을 생성했으나 config 내용은 검사 전과 같았으며 모델 weight는 변경하지 않았다. 이번 수정은 서버의 입력·출력 검증, 평가 도구와 관련 문서에 한정된다.
