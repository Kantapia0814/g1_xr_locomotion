# g1_xr_locomotion

Unitree G1 전신 텔레오퍼레이션: XR로 팔/손 제어, GR00T-WBC로 하체 제어.

이 프로젝트는 **[xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)**(XR 기기를 통한 상체 제어)와 **[GR00T-WholeBodyControl](https://github.com/NVIDIA-Omniverse/GR00T-WholeBodyControl)**(WBC 정책을 통한 하체 보행/밸런스)을 통합하여 **Isaac Sim** 환경에서 Unitree G1 휴머노이드 로봇의 전신 텔레오퍼레이션을 구현합니다.

- **상체** (팔 + 손): Apple Vision Pro 핸드 트래킹을 통한 실시간 제어
- **하체** (다리 + 허리): NVIDIA GR00T Whole-Body-Control 정책 + 키보드 보행 명령
- **통신**: 3개 프로세스가 DDS(CycloneDDS)를 통해 조율, 양쪽 프로젝트의 핵심 로직 수정 없음

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [사전 요구사항](#2-사전-요구사항)
3. [설치](#3-설치)
4. [시뮬레이션 배포](#4-시뮬레이션-배포)
5. [실제 로봇 배포](#5-실제-로봇-배포)
6. [기술 상세](#6-기술-상세)
7. [레포지토리 구조](#7-레포지토리-구조)
8. [감사의 글](#8-감사의-글)

---

## 1. 아키텍처 개요

시스템은 DDS를 통해 통신하는 **3개의 독립 프로세스**로 실행됩니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Apple Vision Pro                             │
│                   (핸드 트래킹 + 헤드 포즈)                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ WebSocket / WebRTC
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  프로세스 3: xr_teleoperate  (conda: tv)                             │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐                       │
│  │ TeleVuer │→ │ Arm IK   │→ │ DDS 발행   │──→ rt/arm_sdk         │
│  │ (VR데이터)│  │ 솔버     │  │            │──→ rt/dex3/left/cmd   │
│  └──────────┘  └──────────┘  │            │──→ rt/dex3/right/cmd  │
│                               └────────────┘                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│ 프로세스 2: GR00T-WBC│  │  프로세스 1: Isaac Sim                    │
│ (conda: gr00t_wbc)   │  │  (conda: unitree_sim_env)                │
│                      │  │                                          │
│ 구독:                │  │  구독:                                    │
│  rt/arm_sdk ─────┐   │  │   rt/lowcmd ──→ 토크 제어 (몸체)         │
│  rt/lowstate     │   │  │   rt/dex3/*  ──→ 위치 제어 (손)          │
│  rt/odostate     │   │  │                                          │
│  rt/secondary_imu│   │  │  발행:                                    │
│                  ▼   │  │   rt/lowstate                            │
│ ┌──────────────────┐ │  │   rt/odostate                           │
│ │ G1GearWbcPolicy  │ │  │   rt/secondary_imu                     │
│ │ (밸런스 + 보행)   │ │  │                                          │
│ │ + Interpolation  │ │  │  카메라 → teleimager → Vision Pro         │
│ │   (팔)           │ │  │                                          │
│ └────────┬─────────┘ │  └──────────────────────────────────────────┘
│          │            │                   ▲
│          ▼            │                   │
│   rt/lowcmd ──────────┼───────────────────┘
│   (29-DOF 통합 명령)   │
└──────────────────────┘

키보드 (wasd): 보행 명령 → GR00T-WBC
```

| 프로세스 | Conda 환경 | 역할 | 제어 대상 |
|----------|-----------|------|----------|
| Isaac Sim | `unitree_sim_env` | 물리 시뮬레이션, 카메라 스트리밍 | 로봇 시뮬레이션 (PhysX) |
| GR00T-WBC | `gr00t_wbc_env` | 전신 보행 + 밸런스 | 다리 (12 DOF) + 허리 (3 DOF) + 팔 (14 DOF) |
| xr_teleoperate | `tv` | XR 핸드 트래킹, 팔 IK, 손 리타겟팅 | 팔 (14 DOF) + 손 (14 DOF) |

---

## 2. 사전 요구사항

### 하드웨어
- **GPU**: NVIDIA RTX GPU (Isaac Sim용, [시스템 요구사항](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html) 참조)
- **XR 기기**: Apple Vision Pro, Meta Quest 3, 또는 PICO 4 Ultra Enterprise
- **라우터**: XR 기기와 호스트 PC를 동일 네트워크에 연결

### 소프트웨어
- Ubuntu 20.04 또는 22.04
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 Anaconda
- NVIDIA Isaac Sim 4.5.0 (Isaac Lab)
- Git

---

## 3. 설치

### 3.1 레포지토리 클론

```bash
git clone https://github.com/Kantapia0814/g1_xr_locomotion.git
cd g1_xr_locomotion
```

### 3.2 셋업 스크립트 실행

셋업 스크립트가 `unitree_sim_isaaclab`을 클론하고 통합 패치를 적용합니다:

```bash
chmod +x setup.sh
./setup.sh
```

> **참고**: `unitree_sim_isaaclab`은 Isaac Lab이 먼저 설치되어야 합니다. [unitree_sim_isaaclab README](https://github.com/unitreerobotics/unitree_sim_isaaclab)를 참조하세요.

### 3.3 Conda 환경 생성 및 패키지 설치

#### 환경 1: Isaac Sim (`unitree_sim_env`)

```bash
# 환경 생성 (Isaac Lab 설치는 unitree_sim_isaaclab README 참조)
conda create -n unitree_sim_env python=3.10
conda activate unitree_sim_env
# Isaac Lab 먼저 설치 (unitree_sim_isaaclab README 참조)
# 그 다음 unitree_sdk2_python 설치:
cd unitree_sdk2_python
pip install -e .
```

#### 환경 2: GR00T-WBC (`gr00t_wbc_env`)

```bash
conda create -n gr00t_wbc_env python=3.10
conda activate gr00t_wbc_env
cd GR00T-WholeBodyControl
pip install -e .
cd ../unitree_sdk2_python
pip install -e .
```

#### 환경 3: xr_teleoperate (`tv`)

```bash
conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate tv

# 서브모듈 설치
cd xr_teleoperate/teleop/teleimager
pip install -e . --no-deps

cd ../televuer
pip install -e .

# SSL 인증서 생성 (XR 기기 연결에 필요)
# Pico / Quest의 경우:
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem

# Apple Vision Pro의 경우 (상세 설정):
openssl genrsa -out rootCA.key 2048
openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 365 -out rootCA.pem -subj "/CN=xr-teleoperate"
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out server.csr -subj "/CN=localhost"
# server_ext.cnf 생성 — IP.2를 본인의 호스트 IP로 변경
cat > server_ext.cnf << 'CERTEOF'
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 192.168.123.164
IP.2 = 192.168.123.2
CERTEOF
openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key \
  -CAcreateserial -out cert.pem -days 365 -sha256 -extfile server_ext.cnf
# rootCA.pem을 Apple Vision Pro에 AirDrop으로 전송 후 설치

# 인증서 경로 설정
mkdir -p ~/.config/xr_teleoperate/
cp cert.pem key.pem ~/.config/xr_teleoperate/

# dex-retargeting 설치
cd ../robot_control/dex-retargeting
pip install -e .

# 나머지 의존성 설치
cd ../../../
pip install -r requirements.txt

# unitree_sdk2_python 설치
cd ../unitree_sdk2_python
pip install -e .
```

---

## 4. 시뮬레이션 배포

### 4.1 실행 순서

3개 프로세스를 **순서대로** 실행해야 합니다. 3개의 별도 터미널을 열어주세요:

#### 터미널 1: Isaac Sim

```bash
conda activate unitree_sim_env
cd unitree_sim_isaaclab

python sim_main.py \
  --task=Isaac-MinimalGround-G129-Dex3-Wholebody \
  --enable_fullbody_dds \
  --enable_cameras \
  --enable_dex3_dds \
  --device cpu \
  --robot_type g129
```

시뮬레이션 창이 나타나고 `controller started, start main loop...`이 출력될 때까지 기다리세요.

> **참고**: 시뮬레이션 창을 한 번 클릭하여 활성화하세요.

#### 터미널 2: GR00T-WBC

```bash
conda activate gr00t_wbc_env
cd GR00T-WholeBodyControl

python -m gr00t_wbc.control.main.teleop.run_g1_xr_control_loop
```

`Waiting for arm commands on rt/arm_sdk from xr_teleoperate...`가 출력될 때까지 기다리세요.

**보행 키보드 조작:**

| 키 | 동작 |
|---|------|
| `]` | WBC 정책 활성화 (밸런스 시작) |
| `o` | 정책 비활성화 |
| `w` / `s` | 전진 / 후진 |
| `a` / `d` | 좌/우 이동 |
| `q` / `e` | 좌/우 회전 |

#### 터미널 3: xr_teleoperate

```bash
conda activate tv
cd xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --sim \
  --arm=G1_29 \
  --ee=dex3 \
  --img-server-ip=localhost
```

> **중요**: 시뮬레이션 모드에서는 `--img-server-ip=localhost`를 사용하세요.

### 4.2 XR 기기 연결

1. XR 헤드셋을 착용하고 호스트 PC와 동일한 Wi-Fi 네트워크에 연결
2. 브라우저를 열고 다음 주소로 이동: `https://<HOST_IP>:8012?ws=wss://<HOST_IP>:8012`
   - `<HOST_IP>`를 호스트 머신의 IP로 변경 (`ifconfig`으로 확인)
   - SSL 인증서 경고가 나오면 수락
3. **Virtual Reality** 클릭 후 모든 프롬프트 허용
4. 팔을 로봇의 초기 포즈에 맞춤 (양팔을 몸 옆으로)
5. 터미널 3에서 **`r`** 키를 눌러 텔레오퍼레이션 시작

### 4.3 추가 옵션

```bash
# 데이터 녹화 포함
python teleop_hand_and_arm.py --sim --arm=G1_29 --ee=dex3 --img-server-ip=localhost --record

# 컨트롤러 트래킹 (핸드 트래킹 대신)
python teleop_hand_and_arm.py --sim --arm=G1_29 --ee=dex3 --img-server-ip=localhost --input-mode=controller

# 헤드리스 모드 (xr_teleoperate 측 GUI 없음)
python teleop_hand_and_arm.py --sim --arm=G1_29 --ee=dex3 --img-server-ip=localhost --headless
```

### 4.4 종료

1. 로봇의 팔을 초기 포즈에 가깝게 위치
2. 터미널 3 (xr_teleoperate)에서 **`q`** 입력
3. 터미널 2 (GR00T-WBC)에서 `Ctrl+C`
4. Isaac Sim 종료

---

## 5. 실제 로봇 배포

실제 배포는 동일한 3-프로세스 아키텍처를 따르되, 다음 차이점이 있습니다:

### 5.1 이미지 서비스 (로봇 PC2에서)

```bash
# 로봇의 개발 컴퓨팅 유닛 (PC2)에서
# teleimager 설치 및 설정: https://github.com/silencht/teleimager
cd ~/teleimager
python -m teleimager.image_server
```

### 5.2 실행

#### 터미널 1: GR00T-WBC (호스트 PC에서)

```bash
conda activate gr00t_wbc_env
cd GR00T-WholeBodyControl

python -m gr00t_wbc.control.main.teleop.run_g1_xr_control_loop \
  --interface=real
```

#### 터미널 2: xr_teleoperate (호스트 PC에서)

```bash
conda activate tv
cd xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm=G1_29 \
  --ee=dex3 \
  --motion \
  --img-server-ip=192.168.123.164
```

> **경고**:
> 1. 항상 로봇으로부터 안전 거리를 유지하세요.
> 2. `--motion` 사용 전 로봇이 제어 모드(R3 리모컨)인지 확인하세요.
> 3. `r` 키를 누르기 전 팔을 초기 포즈에 맞추세요.
> 4. `q` 키를 눌러 종료하기 전 팔을 초기 포즈에 가깝게 위치시키세요.

---

## 6. 기술 상세

### 6.1 GR00T-WBC의 Sim-to-Sim 마이그레이션

GR00T-WholeBodyControl은 원래 자체 **MuJoCo** 시뮬레이터와 함께 실행되도록 설계되었습니다. 우리는 WBC 정책 로직을 보존하면서 MuJoCo를 Isaac Sim으로 대체하는 **sim-to-sim 마이그레이션**을 수행했습니다.

**해결 방법: External Simulator 모드**

새로운 컨트롤 루프(`run_g1_xr_control_loop.py`)를 생성하여 `SIMULATOR = "external"`로 설정, MuJoCo 인스턴스화를 방지합니다. `G1Env`는 실제 로봇과 동일하게 DDS를 통해 Isaac Sim과 통신합니다.

Isaac Sim은 `rt/lowcmd`를 수신하고 내부적으로 임피던스 제어 토크를 계산합니다:

```
torque = tau + kp * (q_target - q_current) + kd * (dq_target - dq_current)
```

**Isaac Sim 추가 수정사항** (`patches/unitree_sim_isaaclab/`):
- `action_provider_wh_dds.py`: MuJoCo 액추에이터 모델에 맞는 전신 토크 제어 모드
- `odo_imu_dds.py`: `rt/odostate` 및 `rt/secondary_imu`용 새 DDS 퍼블리셔
- `g1_29dof_state.py`: GR00T-WBC용 floating-base odometry 및 torso IMU 발행
- `unitree.py`: MuJoCo 값에 맞춘 effort 제한 (허리/발목 50 Nm)
- 카메라 지원이 포함된 새 `Isaac-MinimalGround-G129-Dex3-Wholebody` 태스크

### 6.2 DDS 통신 아키텍처

모든 프로세스 간 통신은 **CycloneDDS**, **Domain ID 1** (시뮬레이션 모드)을 사용합니다.

| DDS 토픽 | 메시지 타입 | 발행자 | 구독자 | 내용 |
|-----------|-----------|--------|--------|------|
| `rt/lowstate` | `LowState_` | Isaac Sim | GR00T-WBC, xr_teleoperate | 29-DOF 관절 상태 + IMU |
| `rt/odostate` | `OdoState_` | Isaac Sim | GR00T-WBC | Floating-base odometry |
| `rt/secondary_imu` | `IMUState_` | Isaac Sim | GR00T-WBC | Torso IMU |
| `rt/lowcmd` | `LowCmd_` | GR00T-WBC | Isaac Sim | 29-DOF 몸체 명령 (q, dq, tau, kp, kd) |
| `rt/arm_sdk` | `LowCmd_` | xr_teleoperate | GR00T-WBC | 14 팔 관절 위치 |
| `rt/dex3/left/cmd` | `HandCmd_` | xr_teleoperate | Isaac Sim | 7-DOF 왼손 명령 |
| `rt/dex3/right/cmd` | `HandCmd_` | xr_teleoperate | Isaac Sim | 7-DOF 오른손 명령 |

**Isaac Sim의 이중 제어 모드:**
- **몸체 관절 (29 DOF)**: 토크 제어 (PD 게인 제로, GR00T-WBC 토크 직접 적용)
- **손 관절 (14 DOF)**: 위치 제어 (PD 게인 유지, xr_teleoperate 위치 타겟)

### 6.3 핵심 통합 과제 및 해결책

**DDS 토픽 충돌 방지**: xr_teleoperate는 팔 명령을 `rt/arm_sdk`로 발행 (`rt/lowcmd`가 아님). GR00T-WBC가 하체 출력과 병합하여 통합 29-DOF 명령을 `rt/lowcmd`로 발행.

**CycloneDDS Fork 안전성**: 손 리타겟팅은 `fork()`을 통한 자식 프로세스에서 실행. CycloneDDS 스레드는 `fork()` 이후 생존하지 않으므로, 메인 프로세스가 공유 메모리에서 읽어 `publish_from_main_process()`를 통해 손 명령을 발행.

**Head Yaw 보상**: VR 헤드셋과 로봇 정면 방향 간 불일치를 보정하기 위해 역 yaw 회전을 손목 위치와 방향에 적용.

**토크 제어 모드에서의 손 명령**: Isaac Sim의 `DDSRLActionProvider`를 수정하여 모든 시뮬레이션 서브스텝에서 몸체 토크 타겟과 함께 손 위치 타겟을 적용.

**GR00T-WBC용 OdoState/IMU**: GR00T-WBC의 `BodyStateProcessor`가 필요로 하는 floating-base odometry 및 torso IMU 데이터를 제공하는 새 `OdoImuDDS` 퍼블리셔 생성.

---

## 7. 레포지토리 구조

```
g1_xr_locomotion/
│
├── README.md                       # 영어 버전
├── README_ko.md                    # 한국어 버전 (이 파일)
├── setup.sh                        # 자동 셋업 (isaaclab 클론 및 패치 적용)
├── .gitignore
│
├── xr_teleoperate/                 # XR을 통한 상체 제어 (수정됨)
│   ├── assets/                     # 로봇 URDF 파일
│   ├── teleop/
│   │   ├── teleop_hand_and_arm.py  # 메인 스크립트 (수정: 손 DDS 재발행)
│   │   ├── robot_control/
│   │   │   ├── robot_hand_unitree.py   # (수정: publish_from_main_process)
│   │   │   ├── robot_arm_ik.py
│   │   │   └── robot_arm.py
│   │   ├── televuer/
│   │   │   └── src/televuer/
│   │   │       └── tv_wrapper.py       # (수정: head yaw 보상)
│   │   ├── teleimager/             # 이미지 스트리밍
│   │   └── utils/                  # 녹화, 필터링, 시각화
│   └── requirements.txt
│
├── GR00T-WholeBodyControl/         # 하체 보행 (수정됨)
│   └── gr00t_wbc/
│       └── control/main/teleop/
│           └── run_g1_xr_control_loop.py   # 신규: Isaac Sim용 XR 컨트롤 루프
│
├── unitree_sdk2_python/            # DDS 통신 라이브러리 (수정됨)
│   └── unitree_sdk2py/idl/
│       ├── default.py              # (수정: OdoState_ 팩토리)
│       └── unitree_hg/msg/dds_/
│           └── _OdoState_.py       # 신규: OdoState IDL 데이터클래스
│
└── patches/
    └── unitree_sim_isaaclab/       # Isaac Sim 패치 (setup.sh가 적용)
        ├── sim_main.py             # (수정: --enable_fullbody_dds 플래그)
        ├── action_provider/
        │   ├── action_provider_wh_dds.py  # (수정: 토크 제어 모드)
        │   └── action_provider_dds.py     # (수정: 전신 관절 매핑)
        ├── dds/
        │   ├── odo_imu_dds.py      # 신규: OdoState + SecondaryIMU 퍼블리셔
        │   ├── dds_create.py       # (수정: OdoImuDDS 등록)
        │   ├── dds_master.py       # (수정: wlo1 인터페이스 바인딩)
        │   └── g1_robot_dds.py     # (수정: IMU 쿼터니언 컨벤션)
        ├── robots/unitree.py       # (수정: MuJoCo에 맞춘 effort 제한)
        └── tasks/                  # (수정 + 신규: MinimalGround wholebody 태스크)
```

---

## 8. 감사의 글

이 프로젝트는 다음 오픈소스 프로젝트를 기반으로 합니다:

- [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) — Unitree XR 텔레오퍼레이션
- [GR00T-WholeBodyControl](https://github.com/NVIDIA-Omniverse/GR00T-WholeBodyControl) — NVIDIA 전신 제어
- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) — Unitree Isaac Lab 시뮬레이션
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) — Unitree Python DDS SDK
- [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) — 텔레오퍼레이션 프레임워크
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — 손 리타겟팅
- [vuer](https://github.com/vuer-ai/vuer) — WebXR 프레임워크
- [pinocchio](https://github.com/stack-of-tasks/pinocchio) — 강체 동역학 라이브러리
