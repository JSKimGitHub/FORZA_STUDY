# Jetson Road Segmentation System 실행 가이드

## 1. 시스템 요구사항

### 하드웨어
- NVIDIA Jetson Nano/Xavier NX/Xavier AGX
- 최소 4GB RAM (8GB 권장)
- 최소 16GB 저장공간

### 소프트웨어
- JetPack 4.6+ 또는 JetPack 5.0+
- Python 3.8+
- CUDA 11.0+

## 2. 설치 방법

### 2.1 자동 설치 (권장)
```bash
# 설치 스크립트 실행
chmod +x install_jetson.sh
./install_jetson.sh
```

### 2.2 수동 설치
```bash
# 시스템 업데이트
sudo apt-get update && sudo apt-get upgrade -y

# 필수 패키지 설치
sudo apt-get install -y python3-pip python3-opencv

# Python 패키지 설치
pip3 install torch torchvision torchaudio numpy opencv-python gymnasium psutil

# Jetson 특화 패키지
pip3 install nvidia-ml-py3 pycuda tensorrt
```

## 3. 실행 방법

### 3.1 기본 실행
```bash
python3 road_segment_jetson.py --video input_video.mp4 --output result.mp4
```

### 3.2 실시간 카메라 입력
```bash
python3 road_segment_jetson.py --video 0 --output realtime_output.mp4
```

### 3.3 성능 최적화 실행
```bash
# GPU 메모리 최대 설정
sudo nvpmodel -m 0
sudo jetson_clocks

# 실행
python3 road_segment_jetson.py --video input.mp4 --output optimized_result.mp4
```

## 4. 성능 최적화 팁

### 4.1 GPU 메모리 관리
- GPU 메모리 사용량을 80% 이하로 유지
- 주기적인 메모리 정리 자동 수행

### 4.2 프레임 처리 최적화
- 프레임 스킵으로 처리 속도 향상
- 이미지 크기 자동 조정 (640px 기준)

### 4.3 멀티스레딩
- 프레임 로딩과 처리를 별도 스레드로 분리
- 큐 기반 비동기 처리

## 5. 문제 해결

### 5.1 CUDA 오류
```bash
# CUDA 버전 확인
nvcc --version

# GPU 상태 확인
nvidia-smi
```

### 5.2 메모리 부족
```bash
# 메모리 사용량 확인
free -h
nvidia-smi

# 스왑 메모리 생성
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 5.3 OpenCV 오류
```bash
# GStreamer 백엔드 설정
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=1
```

## 6. 성능 모니터링

### 6.1 실시간 모니터링
```bash
# GPU 사용량 모니터링
watch -n 1 nvidia-smi

# CPU/메모리 모니터링
htop
```

### 6.2 성능 측정
- FPS (Frames Per Second) 자동 측정
- 프레임별 처리 시간 기록
- 메모리 사용량 추적

## 7. 고급 설정

### 7.1 TensorRT 최적화
```python
# TensorRT 엔진 생성
import tensorrt as trt
# 자동으로 TensorRT 최적화 적용
```

### 7.2 커스텀 파라미터
```bash
# 프레임 스킵 조정
python3 road_segment_jetson.py --video input.mp4 --frame-skip 3

# 메모리 제한 설정
python3 road_segment_jetson.py --video input.mp4 --max-memory 0.7
```

## 8. 출력 파일

### 8.1 비디오 출력
- MP4 형식으로 저장
- 차선 감지 결과 오버레이
- FPS 및 진행률 표시

### 8.2 로그 파일
- 처리 시간 통계
- 메모리 사용량 기록
- 오류 로그

## 9. 추가 기능

### 9.1 DQN 학습
- 강화학습 기반 주행 제어
- 경험 리플레이 버퍼
- 타겟 네트워크 업데이트

### 9.2 차선 변경 감지
- 실시간 차선 변경 패턴 인식
- 변경 이벤트 로깅

## 10. 지원 및 문의

문제가 발생하면 다음을 확인하세요:
1. Jetson 모델 및 JetPack 버전
2. CUDA 및 TensorRT 설치 상태
3. 메모리 사용량
4. 입력 비디오 형식

로그 파일을 확인하여 구체적인 오류 메시지를 확인하세요.

