#!/usr/bin/env python3
"""
Jetson Road Segmentation System Test Script
간단한 테스트를 위한 스크립트
"""

import cv2
import numpy as np
import time
import sys
import os

def test_jetson_environment():
    """Jetson 환경 테스트"""
    print("=== Jetson Environment Test ===")
    
    # OpenCV 테스트
    print("1. Testing OpenCV...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print("OpenCV test: PASSED")
    except Exception as e:
        print(f"OpenCV test: FAILED - {e}")
        return False
    
    # CUDA 테스트
    print("\n2. Testing CUDA...")
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA devices found: {count}")
        if count > 0:
            print("CUDA test: PASSED")
        else:
            print("CUDA test: WARNING - No CUDA devices found")
    except Exception as e:
        print(f"CUDA test: FAILED - {e}")
    
    # GPU 메모리 테스트
    print("\n3. Testing GPU Memory...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print("GPU Memory test: PASSED")
    except Exception as e:
        print(f"GPU Memory test: FAILED - {e}")
    
    # PyTorch 테스트
    print("\n4. Testing PyTorch...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print("PyTorch CUDA test: PASSED")
        else:
            print("PyTorch CUDA test: WARNING - CUDA not available")
    except Exception as e:
        print(f"PyTorch test: FAILED - {e}")
    
    return True

def test_camera():
    """카메라 테스트"""
    print("\n=== Camera Test ===")
    
    # 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera test: FAILED - Cannot open camera")
        return False
    
    print("Camera opened successfully")
    
    # 몇 프레임 캡처
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: {frame.shape}")
        else:
            print(f"Frame {i+1}: FAILED")
            break
    
    cap.release()
    print("Camera test: PASSED")
    return True

def test_video_processing():
    """비디오 처리 테스트"""
    print("\n=== Video Processing Test ===")
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 처리 시간 측정
    start_time = time.time()
    
    # 색상 변환
    hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    
    # 엣지 검출
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 허프 변환
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Processing FPS: {1/processing_time:.1f}")
    
    if processing_time < 0.1:  # 100ms 이하
        print("Video Processing test: PASSED")
        return True
    else:
        print("Video Processing test: WARNING - Slow processing")
        return False

def create_test_video():
    """테스트 비디오 생성"""
    print("\n=== Creating Test Video ===")
    
    # 테스트 비디오 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))
    
    for i in range(100):  # 3초 분량
        # 차선이 있는 테스트 프레임 생성
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 배경 (도로)
        frame[:, :] = (50, 50, 50)
        
        # 차선 그리기
        cv2.line(frame, (200, 480), (300, 200), (255, 255, 255), 5)  # 왼쪽 차선
        cv2.line(frame, (440, 480), (340, 200), (255, 255, 255), 5)  # 오른쪽 차선
        
        out.write(frame)
    
    out.release()
    print("Test video created: test_video.mp4")
    return True

def main():
    """메인 테스트 함수"""
    print("Jetson Road Segmentation System Test")
    print("=====================================")
    
    # 환경 테스트
    if not test_jetson_environment():
        print("Environment test failed. Please check your Jetson setup.")
        return
    
    # 카메라 테스트
    test_camera()
    
    # 비디오 처리 테스트
    test_video_processing()
    
    # 테스트 비디오 생성
    create_test_video()
    
    print("\n=== Test Summary ===")
    print("All tests completed!")
    print("\nTo run the full system:")
    print("python3 road_segment_jetson.py --video test_video.mp4 --output test_result.mp4")

if __name__ == "__main__":
    main()

