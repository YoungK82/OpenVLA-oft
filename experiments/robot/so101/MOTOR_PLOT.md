# 실시간 모터 그래프

`plot_motor_feedback.py`는 CSV만 읽는 별도 Python 프로그램입니다.
로봇 제어·모터 설정·LeRobot 코드를 변경하거나 모터에 직접 접속하지 않습니다.
기존 `openvla-oft` 가상환경에서 실행합니다. 별도 가상환경은 필요하지 않습니다.

## 실행

프로젝트 폴더에서 로봇 평가를 기존 방식으로 시작한 뒤 **다른 터미널**에서 실행합니다.
새 터미널에서도 먼저 `conda activate openvla-oft`를 실행하세요.
아래 명령의 `python`은 활성화된 `openvla-oft` 환경의 Python입니다.

```powershell
python experiments/robot/so101/plot_motor_feedback.py --latest experiments/logs
```

`--latest`는 시작할 때 가장 최근에 수정된 `MOTOR-FEEDBACK-*.csv` 하나를 선택합니다.
새 실험 파일로 자동 전환하지 않습니다. 터미널에 선택한 경로가 표시됩니다.
특정 파일을 지정하려면 다음 명령을 사용합니다. 파일이 아직 없으면 생성될 때까지 기다립니다.

```powershell
python experiments/robot/so101/plot_motor_feedback.py --csv 'experiments/logs/MOTOR-FEEDBACK-실행시각.csv'
```

## 표시와 종료

- 위: 전류(A), 아래: 부호 있는 PWM 출력 비율(%). 6개 모터의 색상은 두 그래프에서 동일합니다.
- 가로축은 CSV 최초 측정부터의 경과 시간입니다. 기본 최근 30초, 200ms마다 갱신합니다.
- 우측 체크박스로 모터를 선택합니다. `Pause view`로 화면을 멈춘 뒤 도구 모음으로 확대·이동·이미지 저장할 수 있습니다.
- 일시정지 중에도 CSV 읽기는 계속하며, 재개하면 최근 구간을 표시합니다.
- `--torque`는 **미보정 토크 크기 추정값**을 세 번째 그래프에 추가합니다.
- `--window 60 --interval-ms 500`으로 최근 60초·0.5초 갱신을 선택할 수 있습니다.
- 오류로 빈 값은 0으로 바꾸지 않고 선을 끊습니다. 에피소드 변경이나 누락 step도 선을 끊습니다.
- 아래쪽에 마지막 수신 이후 시간, 에피소드/step, 읽은 행 수, 오류 행 수와 최근 오류를 표시합니다.
- **로봇 실행이 끝나도 창은 자동 종료하지 않습니다.** 데이터가 없으면 추론 대기인지 종료인지 알 수 없으므로 마지막 그래프를 유지합니다.
- 창의 X 버튼 또는 그래프 터미널의 `Ctrl+C`로 뷰어를 종료합니다. 로봇 평가에는 종료 명령을 보내지 않습니다.

화면 갱신 주기는 모터 측정 주기가 아닙니다. 갱신 사이 도착한 행은 모두 반영합니다.
추론을 기다리는 동안 현재 로거는 추가 측정을 하지 않습니다.
화면용 메모리는 시간 구간과 모터별 최대 20,000점으로 제한합니다. 매우 큰 기존 CSV는
여러 갱신에 걸쳐 읽으며, 화면 갱신마다 최대 256KiB를 처리합니다.
측정 원본은 CSV에 그대로 남습니다. 긴 레코드는 완성될 때까지 기다리고,
비정상적으로 1MiB를 넘으면 오류로 중단합니다.

## 종료된 실험 확인

전체 구간을 표시하고 추가 행을 감시하지 않으려면 `--offline`을 사용합니다.

```powershell
python experiments/robot/so101/plot_motor_feedback.py --latest experiments/logs --offline --torque
```

창 없이 그림 파일을 저장하려면:

```powershell
python experiments/robot/so101/plot_motor_feedback.py --latest experiments/logs --offline --torque --save outputs/motor-feedback.png
```

## 설치와 테스트

로봇 없이 화면을 먼저 확인할 수 있는 **합성 데이터**를
`outputs/motor-feedback-demo/MOTOR-FEEDBACK-DEMO.csv`에 준비했습니다.
`outputs/motor-feedback-demo/preview.png`도 합성 데이터로 만든 예시 그림입니다.

```powershell
python experiments/robot/so101/plot_motor_feedback.py --csv outputs/motor-feedback-demo/MOTOR-FEEDBACK-DEMO.csv --offline --torque
```

`openvla-oft` 환경에 그래프 라이브러리를 설치할 때:

```powershell
conda activate openvla-oft
python -m pip install -r experiments/robot/so101/requirements_motor_plot.txt
```

그래프 창을 표시하려면 Tk 또는 Qt 같은 GUI 백엔드가 필요합니다. 이 PC는 기존
Python 배포판의 Tk를 활용합니다. 로봇 평가와 뷰어는 같은 환경의 별도 프로세스로 실행됩니다.

가짜 CSV 기반 테스트(실제 모터 구동 없음):

```powershell
python -B -m unittest experiments.robot.so101.test_plot_motor_feedback -v
```
