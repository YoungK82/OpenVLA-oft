# 모터 피드백 CSV

실시간 그래프와 종료 후 분석은 [MOTOR_PLOT.md](MOTOR_PLOT.md)를 참고하세요.

`run_so101_eval.py` 실행 시 기존 평가 로그와 같은 폴더에
`MOTOR-FEEDBACK-<실행시각><note>.csv`를 기본 생성합니다.
`--log_motor_feedback False`를 추가하면 CSV 생성과 추가 버스 읽기를 끕니다.
외부 LeRobot 설치 파일이나 모터 레지스터 설정을 변경하지 않습니다.

기본 CSV는 아래 **8개 필드만** 저장합니다.

| 필드 | 의미 |
|---|---|
| `episode` | 에피소드 번호, 1부터 시작 |
| `step` | 에피소드 내 제어 단계, 0부터 시작 |
| `motor` | 관절 이름. 각 step마다 6개 모터를 한 행씩 기록 |
| `timestamp_unix_s` | 해당 step의 피드백 읽기 시작 UNIX 시각(초, 소수 포함) |
| `current_A` | 전류 원시값 × 0.0065, 단위 A |
| `load_pct` | LeRobot이 부호를 해석한 부하값 / 10, 방향을 포함한 PWM 출력 비율 (%) |
| `errors` | 읽기 실패·누락·유효하지 않은 값의 설명. 정상일 때 빈칸 |
| `torque_est_Nm` | 미보정 출력축 토크 크기 추정치, 단위 N·m |

원시값은 환산에만 사용하고 저장하지 않습니다. 위치·속도와 세부 읽기 시간도
저장하지 않습니다. 기존 위치 관측은 정책 실행을 위해 유지하며, 추가 속도 읽기는 없습니다.

토크 계산식은 다음과 같습니다.

```text
current_A = present_current_raw × 0.0065
load_pct = present_load / 10
torque_est_Nm = current_A × 11.0 × 0.0980665
```

`--torque_constant_kgf_cm_per_A 11.0`으로 토크 상수를 바꿀 수 있습니다.
12V 30kgf·cm 모터의 **출력축 기준 상수를 피드백 전류에 적용할 수 있다는 가정**이며,
기어비 345를 다시 곱하지 않습니다. 토크 방향은 추론하지 않으며,
실측 토크나 보호 판단용으로 검증된 값이 아닙니다.

ST 계열 문서의 주소 69는 2바이트, 단위 6.5mA, 모터 상전류 피드백으로 정의됩니다:
https://docs.waveshare.net/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation/

매 제어 step에서 `get_observation()` 이후, 추론 요청/행동 전송 전에 전류·부하를
순서대로 읽습니다. 같은 step의 6행은 동일한 시작 시각을 갖지만, 실제 센서 내부의
동시 측정 시각을 뜻하지는 않습니다. 추론 서버를 기다리는 동안 백그라운드로
수집하지 않습니다. dry-run에서도 피드백은 읽으며 실행 모드는 기존 평가 로그에서 확인합니다.

추가 읽기는 2회 통신이며 재시도하지 않습니다. 통신 시간이 추가되므로 30Hz 유지
여부는 하드웨어에서 확인해야 합니다. 읽기 실패 시 해당 값과 그 파생값은 빈칸으로
남기고 `errors`에 기록하며 제어를 계속합니다. 이전 값이나 0으로 대체하지 않습니다.
각 step을 기록한 뒤 flush하며, 저장 실패는 기존 종료 처리로 전달합니다.

오프라인 검증(로봇/서버 연결 없음):

```powershell
& 'C:\Users\young\miniconda3\envs\openvla-oft\python.exe' -B -m unittest experiments.robot.so101.test_motor_feedback -v
```
