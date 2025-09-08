#!/usr/bin/env python3
"""
Jetson Optimized Road Segmentation and Autonomous Driving System
Optimized for NVIDIA Jetson devices with TensorRT acceleration
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import time
import threading
from queue import Queue
import os
import sys
import argparse

# Jetson specific imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    JETSON_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT/CUDA not available. Running in CPU mode.")
    JETSON_AVAILABLE = False

# GPU 사용 가능 여부 확인
if torch.cuda.is_available() and JETSON_AVAILABLE:
    device = torch.device("cuda")
    print("GPU mode enabled")
else:
    device = torch.device("cpu")
    print("CPU mode enabled")

# Memory management for Jetson
import gc
import psutil

class JetsonMemoryManager:
    """Jetson GPU 메모리 관리 클래스"""
    
    def __init__(self):
        self.gpu_memory_usage = 0
        self.max_gpu_memory = 0.8  # 80% 사용 제한
        
    def check_memory(self):
        """메모리 사용량 체크"""
        if JETSON_AVAILABLE:
            try:
                # GPU 메모리 사용량 체크
                gpu_memory = cuda.mem_get_info()
                available_memory = gpu_memory[0] / (1024**3)  # GB
                total_memory = gpu_memory[1] / (1024**3)  # GB
                usage_percent = (total_memory - available_memory) / total_memory
                
                if usage_percent > self.max_gpu_memory:
                    print(f"Warning: GPU memory usage high: {usage_percent:.2%}")
                    self.cleanup_memory()
                    
            except Exception as e:
                print(f"GPU memory check failed: {e}")
        
        # CPU 메모리 체크
        cpu_memory = psutil.virtual_memory()
        if cpu_memory.percent > 80:
            print(f"Warning: CPU memory usage high: {cpu_memory.percent:.1f}%")
            self.cleanup_memory()
    
    def cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        if JETSON_AVAILABLE:
            try:
                cuda.Context.pop()
                cuda.Context.push()
            except:
                pass

class JetsonOptimizedLaneDetector:
    """Jetson 최적화 차선 감지기"""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prev_lane_state = "center"
        self.change_counter = 0
        self.threshold_frames = 5
        self.margin = 50
        self.vanish_history = []
        self.lane_width_history = []
        
        # 검출 상태
        self.left_detect = False
        self.right_detect = False
        self.left_m = self.right_m = 0
        self.left_b = self.right_b = (0, 0)
        
        # 이전 프레임 lane 좌표 저장
        self.prev_lanes = [None, None]
        
        # Jetson 최적화 설정
        self.frame_skip = 2  # 프레임 스킵으로 처리 속도 향상
        self.frame_count = 0
        
    def filter_colors(self, image):
        """색상 필터링 (Jetson 최적화)"""
        # 이미지 크기 축소로 처리 속도 향상
        height, width = image.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 흰색 차선
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 노란색 차선
        lower_yellow = np.array([18, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 마스크 결합
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        return combined_mask
    
    def limit_region(self, image):
        """ROI 설정"""
        height, width = image.shape[:2]
        mask = np.zeros_like(image)
        
        # 관심 영역 정의 (다각형)
        polygon = np.array([
            [(0, height),
             (width//2, int(height*0.55)),
             (width//2, int(height*0.45)),
             (width, int(height*0.55)),
             (width, height)]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [polygon], 255)
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def houghLines(self, image):
        """허프 직선 검출 (최적화)"""
        return cv2.HoughLinesP(
            image, 
            1, 
            np.pi/180, 
            40, 
            minLineLength=30, 
            maxLineGap=120
        )
    
    def separateLine(self, image, lines):
        """좌우 차선 분리"""
        left, right = [], []
        height, width = image.shape[:2]
        self.img_center = width // 2
        slope_thresh = 0.3
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                if abs(slope) < slope_thresh:
                    continue
                    
                if slope < 0:  # 왼쪽 차선
                    left.append(line[0])
                else:  # 오른쪽 차선
                    right.append(line[0])
        
        return left, right
    
    def regression(self, separated, image):
        """회귀선 계산"""
        left, right = separated
        height = image.shape[0]
        lanes = []
        
        # 오른쪽 차선
        if len(right) > 0:
            right_x = [line[0] for line in right] + [line[2] for line in right]
            right_y = [line[1] for line in right] + [line[3] for line in right]
            
            if len(right_x) > 1:
                right_coeff = np.polyfit(right_y, right_x, 1)
                right_m, right_b = right_coeff
                
                # 차선의 시작과 끝점 계산
                right_start = (int(right_m * height + right_b), height)
                right_end = (int(right_m * (height * 0.6) + right_b), int(height * 0.6))
                
                lanes.append((right_start, right_end))
                self.right_m, self.right_b = right_m, right_b
                self.right_detect = True
            else:
                lanes.append(self.prev_lanes[0])
        else:
            lanes.append(self.prev_lanes[0])
        
        # 왼쪽 차선
        if len(left) > 0:
            left_x = [line[0] for line in left] + [line[2] for line in left]
            left_y = [line[1] for line in left] + [line[3] for line in left]
            
            if len(left_x) > 1:
                left_coeff = np.polyfit(left_y, left_x, 1)
                left_m, left_b = left_coeff
                
                # 차선의 시작과 끝점 계산
                left_start = (int(left_m * height + left_b), height)
                left_end = (int(left_m * (height * 0.6) + left_b), int(height * 0.6))
                
                lanes.append((left_start, left_end))
                self.left_m, self.left_b = left_m, left_b
                self.left_detect = True
            else:
                lanes.append(self.prev_lanes[1])
        else:
            lanes.append(self.prev_lanes[1])
        
        self.prev_lanes = lanes
        return lanes
    
    def detect_lane_change(self, lanes):
        """차선 변경 감지"""
        if len(lanes) < 2:
            return "center"
        
        left_lane, right_lane = lanes
        
        if left_lane is None or right_lane is None:
            return "center"
        
        # 차선 중앙점 계산
        left_center = (left_lane[0][0] + left_lane[1][0]) // 2
        right_center = (right_lane[0][0] + right_lane[1][0]) // 2
        lane_center = (left_center + right_center) // 2
        
        # 현재 차선 상태 결정
        if abs(lane_center - self.img_center) < self.margin:
            current_state = "center"
        elif lane_center < self.img_center:
            current_state = "left"
        else:
            current_state = "right"
        
        # 차선 변경 감지
        if current_state != self.prev_lane_state:
            self.change_counter += 1
            if self.change_counter >= self.threshold_frames:
                self.prev_lane_state = current_state
                return f"change_to_{current_state}"
        else:
            self.change_counter = 0
        
        return current_state
    
    def process_frame(self, frame):
        """프레임 처리 (Jetson 최적화)"""
        # 프레임 스킵으로 처리 속도 향상
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return None, "center"
        
        # 메모리 체크
        self.memory_manager.check_memory()
        
        # 차선 감지 파이프라인
        filtered = self.filter_colors(frame)
        roi = self.limit_region(filtered)
        edges = cv2.Canny(roi, 50, 150)
        lines = self.houghLines(edges)
        separated = self.separateLine(edges, lines)
        lanes = self.regression(separated, frame)
        lane_state = self.detect_lane_change(lanes)
        
        return lanes, lane_state

    def process_frame_noskip(self, frame):
        """프레임 처리 (스킵/메모리체크 없이 점수 계산용)"""
        filtered = self.filter_colors(frame)
        roi = self.limit_region(filtered)
        edges = cv2.Canny(roi, 50, 150)
        lines = self.houghLines(edges)
        separated = self.separateLine(edges, lines)
        lanes = self.regression(separated, frame)
        lane_state = self.detect_lane_change(lanes)
        return lanes, lane_state, edges

class JetsonOptimizedDQN(nn.Module):
    """Jetson 최적화 DQN"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 과적합 방지
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim)
        )
        
        # Jetson 최적화
        if JETSON_AVAILABLE:
            self.cuda()
    
    def forward(self, x):
        return self.fc(x)

class SimSiamEncoder(nn.Module):
    """경량 SimSiam 스타일 SSL 인코더 (Jetson 친화적)"""
    def __init__(self, feature_dim=128):
        super().__init__()
        # 경량 백본 CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # 프로젝트 헤드 (MLP)
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )
        # 프리딕터 헤드 (MLP)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )
        if JETSON_AVAILABLE and torch.cuda.is_available():
            self.cuda()

    def encode(self, x):
        z = self.backbone(x)
        z = z.view(z.size(0), -1)
        z = self.projector(z)
        return z

    def forward(self, x):
        # 반환은 인코더 임베딩
        return self.encode(x)

def _negative_cosine_similarity(p, z):
    # SimSiam loss: -cosine(p, z) with stop-grad on z
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

class SSLPretrainer:
    """프레임에 대한 SimSiam 사전학습 루프"""
    def __init__(self, input_size=(128, 128), feature_dim=128, lr=1e-3):
        self.input_w = input_size[0]
        self.input_h = input_size[1]
        self.model = SimSiamEncoder(feature_dim=feature_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if (JETSON_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)

    def _augment(self, imgs):
        # 간단한 색상/블러/크롭 증강 (numpy -> torch)
        batch = []
        for img in imgs:
            h, w = img.shape[:2]
            # 랜덤 크롭 (가벼운)
            crop_scale = random.uniform(0.8, 1.0)
            ch, cw = int(h * crop_scale), int(w * crop_scale)
            y0 = random.randint(0, max(0, h - ch))
            x0 = random.randint(0, max(0, w - cw))
            cropped = img[y0:y0+ch, x0:x0+cw]
            # 리사이즈
            resized = cv2.resize(cropped, (self.input_w, self.input_h))
            # 색상 지터 (밝기/대비)
            alpha = random.uniform(0.8, 1.2)  # 대비
            beta = random.uniform(-20, 20)    # 밝기
            jitter = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
            # 가우시안 블러 (확률)
            if random.random() < 0.3:
                k = random.choice([3, 5])
                jitter = cv2.GaussianBlur(jitter, (k, k), 0)
            batch.append(jitter)
        batch = np.stack(batch, axis=0)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        return batch.to(self.device)

    def train_epoch(self, frames, batch_size=32):
        self.model.train()
        num_frames = len(frames)
        if num_frames == 0:
            return 0.0
        indices = np.random.permutation(num_frames)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            idx = indices[start:end]
            imgs = [frames[i] for i in idx]
            v1 = self._augment(imgs)
            v2 = self._augment(imgs)
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                z1 = self.model.encode(v1)
                z2 = self.model.encode(v2)
                p1 = self.model.predictor(z1)
                p2 = self.model.predictor(z2)
                loss = _negative_cosine_similarity(p1, z2) / 2.0 + _negative_cosine_similarity(p2, z1) / 2.0
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(1, num_batches)

    def extract_features(self, frames, batch_size=64):
        self.model.eval()
        features = []
        with torch.no_grad():
            for start in range(0, len(frames), batch_size):
                batch = frames[start:start+batch_size]
                x = self._augment(batch)  # 증강 없이 리사이즈/정규화를 위해 _augment의 경로 재사용 (jitter 최소화)
                # 증강을 끄기 위해 다시 리사이즈만 수행
                # 위 augment가 색상/블러를 넣으므로 여기서는 별도 경로를 사용
                x_np = []
                for img in batch:
                    resized = cv2.resize(img, (self.input_w, self.input_h))
                    x_np.append(resized)
                x = torch.from_numpy(np.stack(x_np, axis=0)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                z = self.model.encode(x)
                features.append(z.cpu().numpy())
        if len(features) == 0:
            return []
        return np.concatenate(features, axis=0)

class JetsonDrivingEnv(gym.Env):
    """Jetson 최적화 주행 환경"""
    
    def __init__(self, frames, lane_detector, memory_manager, ssl_features=None, use_ssl_features=False):
        super().__init__()
        if len(frames) == 0:
            raise ValueError("frames empty")
        
        self.frames = frames
        self.current_idx = 0
        self.env_h, self.env_w = frames[0].shape[:2]
        self.car_x = self.env_w // 2
        self.lane_detector = lane_detector
        self.memory_manager = memory_manager
        self.ssl_features = ssl_features
        self.use_ssl_features = use_ssl_features and (ssl_features is not None) and (len(ssl_features) == len(frames))
        self.ssl_feature_dim = int(ssl_features.shape[1]) if (ssl_features is not None and hasattr(ssl_features, 'shape')) else 0
        
        # 행동 공간: 0=좌회전, 1=직진, 2=우회전
        self.action_space = spaces.Discrete(3)
        
        # 상태 공간: 차선 위치, 차량 위치 등
        obs_dim = 4 + (self.ssl_feature_dim if self.use_ssl_features else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 성능 모니터링
        self.step_count = 0
        self.reward_history = []
        
    def reset(self, seed=None):
        """환경 리셋"""
        super().reset(seed=seed)
        self.current_idx = 0
        self.step_count = 0
        return self._get_state(), {}
    
    def _get_state(self):
        """현재 상태 반환"""
        if self.current_idx >= len(self.frames):
            dim = 4 + (self.ssl_feature_dim if self.use_ssl_features else 0)
            return np.zeros(dim, dtype=np.float32)
        
        frame = self.frames[self.current_idx]
        lanes, lane_state = self.lane_detector.process_frame(frame)
        
        # 상태 벡터 구성
        state = np.zeros(4, dtype=np.float32)
        
        if lanes and len(lanes) >= 2:
            left_lane, right_lane = lanes
            
            if left_lane is not None:
                left_center = (left_lane[0][0] + left_lane[1][0]) / 2
                state[0] = left_center / self.env_w  # 정규화
            else:
                state[0] = 0.0
            
            if right_lane is not None:
                right_center = (right_lane[0][0] + right_lane[1][0]) / 2
                state[1] = right_center / self.env_w  # 정규화
            else:
                state[1] = 1.0
            
            # 차선 중앙
            if left_lane is not None and right_lane is not None:
                lane_center = (left_center + right_center) / 2
                state[2] = lane_center / self.env_w
            else:
                state[2] = 0.5
        
        # 차량 위치
        state[3] = self.car_x / self.env_w
        
        if self.use_ssl_features and self.ssl_features is not None:
            feat = self.ssl_features[self.current_idx].astype(np.float32)
            # 정규화된 연속 특징이므로 그대로 연결
            state = np.concatenate([state, feat], axis=0)
        
        return state
    
    def step(self, action):
        """한 스텝 진행"""
        self.step_count += 1
        
        # 행동에 따른 차량 위치 업데이트
        if action == 0:  # 좌회전
            self.car_x = max(0, self.car_x - 10)
        elif action == 2:  # 우회전
            self.car_x = min(self.env_w, self.car_x + 10)
        
        # 다음 프레임으로 이동
        self.current_idx += 1
        
        # 상태 획득
        state = self._get_state()
        
        # 보상 계산
        reward = self._calculate_reward(state, action)
        self.reward_history.append(reward)
        
        # 종료 조건
        done = self.current_idx >= len(self.frames) - 1
        
        # 정보
        info = {
            'step': self.step_count,
            'frame_idx': self.current_idx,
            'reward': reward
        }
        
        return state, reward, done, False, info
    
    def _calculate_reward(self, state, action):
        """보상 계산"""
        reward = 0.0
        
        # 차선 중앙 유지 보상
        lane_center = state[2]
        car_position = state[3]
        
        # 차선 중앙에 가까울수록 높은 보상
        distance_from_center = abs(car_position - lane_center)
        if distance_from_center < 0.1:
            reward += 10.0
        elif distance_from_center < 0.2:
            reward += 5.0
        else:
            reward -= 5.0
        
        # 차선 이탈 페널티
        if distance_from_center > 0.4:
            reward -= 20.0
        
        # 안정적인 주행 보상
        reward += 1.0
        
        return reward

class JetsonRoadSegmentationSystem:
    """Jetson 최적화 도로 분할 시스템"""
    
    def __init__(self, video_path=None, output_path="output_jetson.mp4",
                 feature_dim=128, ssl_input_size=(128, 128)):
        self.video_path = video_path
        self.output_path = output_path
        self.memory_manager = JetsonMemoryManager()
        self.feature_dim = feature_dim
        self.ssl_input_size = ssl_input_size
        
        # 성능 모니터링
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.processing_times = []
        
        # 멀티스레딩 설정
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.running = False
        
        print("Jetson Road Segmentation System Initialized")
        print(f"TensorRT Available: {JETSON_AVAILABLE}")

    def pretrain_ssl(self, frames, epochs=10, batch_size=32, lr=1e-3):
        """SSL 사전학습 수행 및 특징 추출"""
        print("Starting SSL pretraining (SimSiam)...")
        pretrainer = SSLPretrainer(input_size=self.ssl_input_size, feature_dim=self.feature_dim, lr=lr)
        for ep in range(epochs):
            loss = pretrainer.train_epoch(frames, batch_size=batch_size)
            if (ep + 1) % 1 == 0:
                print(f"SSL Epoch {ep+1}/{epochs} - Loss: {loss:.4f}")
        feats = pretrainer.extract_features(frames, batch_size=64)
        print(f"SSL pretraining completed. Feature shape: {np.shape(feats)}")
        return feats, pretrainer.model

class ContinualLearningBuffer:
    """연속학습을 위한 경험 리플레이 버퍼"""
    
    def __init__(self, max_size=50000, task_memory_ratio=0.1):
        self.max_size = max_size
        self.task_memory_ratio = task_memory_ratio  # 각 태스크당 보존할 메모리 비율
        self.buffers = {}  # task_id -> deque
        self.task_sizes = {}  # task_id -> size
        
    def add_experience(self, state, action, reward, next_state, done, task_id):
        """경험 추가"""
        if task_id not in self.buffers:
            self.buffers[task_id] = deque(maxlen=int(self.max_size * self.task_memory_ratio))
            self.task_sizes[task_id] = 0
        
        self.buffers[task_id].append((state, action, reward, next_state, done))
        self.task_sizes[task_id] += 1
    
    def sample_batch(self, batch_size, current_task_id, replay_ratio=0.3):
        """현재 태스크 + 이전 태스크들에서 배치 샘플링"""
        current_experiences = list(self.buffers.get(current_task_id, []))
        replay_experiences = []
        
        # 이전 태스크들에서 경험 리플레이
        for task_id, buffer in self.buffers.items():
            if task_id != current_task_id and len(buffer) > 0:
                replay_experiences.extend(list(buffer))
        
        # 현재 태스크와 리플레이 비율로 샘플링
        current_size = int(batch_size * (1 - replay_ratio))
        replay_size = batch_size - current_size
        
        batch = []
        if len(current_experiences) > 0:
            batch.extend(random.sample(current_experiences, min(current_size, len(current_experiences))))
        if len(replay_experiences) > 0:
            batch.extend(random.sample(replay_experiences, min(replay_size, len(replay_experiences))))
        
        return batch
    
    def get_task_count(self):
        return len(self.buffers)

class ElasticWeightConsolidation:
    """EWC (Elastic Weight Consolidation) for catastrophic forgetting prevention"""
    
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optimal_params = {}
        
    def compute_fisher_info(self, dataloader, num_samples=1000):
        """Fisher Information Matrix 계산"""
        self.model.eval()
        fisher_info = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            self.model.zero_grad()
            # 간단한 forward pass로 gradient 계산
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                states = batch[0]
                if len(states) > 0:
                    outputs = self.model(states)
                    # Fisher 정보는 gradient의 제곱
                    loss = outputs.sum()
                    loss.backward()
                    
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            fisher_info[name] += param.grad.data ** 2
                    sample_count += len(states)
        
        # 정규화
        for name in fisher_info:
            fisher_info[name] /= max(sample_count, 1)
        
        self.fisher_info = fisher_info
        
    def save_optimal_params(self):
        """현재 모델 파라미터를 최적 파라미터로 저장"""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
    
    def ewc_loss(self):
        """EWC 정규화 손실 계산"""
        if not self.fisher_info or not self.optimal_params:
            return 0
        
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                ewc_loss += (self.fisher_info[name] * (param - self.optimal_params[name]) ** 2).sum()
        
        return self.lambda_ewc * ewc_loss
    
    def _load_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        file_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            file_frames.append(frame)
        cap.release()
        return file_frames

    def load_video(self, video_path):
        """단일 파일 또는 디렉토리의 모든 비디오에서 프레임 로드"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")
        frames = []
        if os.path.isdir(video_path):
            print(f"Loading videos from directory: {video_path}")
            exts = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
            files = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path)) if os.path.splitext(f)[1] in exts]
            if len(files) == 0:
                print("No video files found in directory.")
            for fp in files:
                print(f"Loading: {os.path.basename(fp)}")
                f = self._load_video_file(fp)
                frames.extend(f)
                print(f"Loaded {len(f)} frames from {os.path.basename(fp)} (total {len(frames)})")
        else:
            print(f"Loading video frames from file: {video_path}")
            frames = self._load_video_file(video_path)
        print(f"Total frames loaded: {len(frames)}")
        return frames
    
    def train_dqn_continual(self, video_paths, epochs_per_task=50, use_ssl_features=False,
                           lambda_ewc=1000, replay_ratio=0.3, task_memory_ratio=0.1):
        """연속학습 DQN - 여러 비디오/태스크에 대해 순차적 학습"""
        print("Starting Continual Learning DQN training...")
        
        # 상태 차원 결정 (SSL 특징 포함)
        ssl_dim = self.feature_dim if use_ssl_features else 0
        state_dim = 4 + ssl_dim
        action_dim = 3
        
        # 네트워크 초기화
        policy_net = JetsonOptimizedDQN(state_dim, action_dim)
        target_net = JetsonOptimizedDQN(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        
        optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
        
        # 연속학습 컴포넌트
        continual_buffer = ContinualLearningBuffer(task_memory_ratio=task_memory_ratio)
        ewc = ElasticWeightConsolidation(policy_net, lambda_ewc=lambda_ewc)
        
        # 학습 파라미터
        gamma = 0.99
        epsilon_start = 0.2
        epsilon_end = 0.05
        batch_size = 32
        update_frequency = 10
        
        all_episode_rewards = []
        task_id = 0
        
        for video_path in video_paths:
            print(f"\n=== Training on Task {task_id}: {video_path} ===")
            
            # 현재 태스크 데이터 로드
            frames = self.load_video(video_path)
            if len(frames) == 0:
                print(f"No frames loaded from {video_path}, skipping...")
                continue
            
            # SSL 특징 추출 (필요시)
            ssl_features = None
            if use_ssl_features:
                ssl_features, _ = self.pretrain_ssl(frames, epochs=5)
            
            # 환경 초기화
            lane_detector = JetsonOptimizedLaneDetector(self.memory_manager)
            env = JetsonDrivingEnv(frames, lane_detector, self.memory_manager,
                                 ssl_features=ssl_features, use_ssl_features=use_ssl_features)
            
            # EWC를 위한 Fisher 정보 계산 (이전 태스크가 있는 경우)
            if task_id > 0:
                print("Computing Fisher Information for EWC...")
                # 간단한 데이터로더 생성
                dummy_states = []
                for i in range(0, len(frames), 10):  # 샘플링
                    state, _ = env.reset()
                    env.current_idx = i
                    state = env._get_state()
                    dummy_states.append(torch.tensor(state, dtype=torch.float32))
                
                if len(dummy_states) > 0:
                    dummy_loader = [(torch.stack(dummy_states),)]
                    ewc.compute_fisher_info(dummy_loader)
            
            # 현재 태스크 학습
            epsilon = epsilon_start
            task_episode_rewards = []
            
            for episode in range(epochs_per_task):
                state, _ = env.reset()
                total_reward = 0
                step_count = 0
                
                while True:
                    # ε-greedy 정책
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            q_values = policy_net(torch.tensor(state, dtype=torch.float32))
                            action = q_values.argmax().item()
                    
                    # 환경에서 한 스텝 진행
                    next_state, reward, done, _, _ = env.step(action)
                    total_reward += reward
                    step_count += 1
                    
                    # 연속학습 버퍼에 경험 저장
                    continual_buffer.add_experience(state, action, reward, next_state, done, task_id)
                    
                    # 배치 학습 (현재 + 이전 태스크 경험 리플레이)
                    if continual_buffer.get_task_count() > 0:
                        batch = continual_buffer.sample_batch(batch_size, task_id, replay_ratio)
                        
                        if len(batch) > 0:
                            states = torch.tensor([exp[0] for exp in batch], dtype=torch.float32)
                            actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
                            rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
                            next_states = torch.tensor([exp[3] for exp in batch], dtype=torch.float32)
                            dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
                            
                            # Q-러닝 업데이트
                            current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                            
                            with torch.no_grad():
                                next_q_values = target_net(next_states).max(1)[0]
                                target_q_values = rewards + gamma * (1 - dones.float()) * next_q_values
                            
                            # 기본 손실
                            q_loss = nn.MSELoss()(current_q_values, target_q_values)
                            
                            # EWC 정규화 손실 추가
                            ewc_loss = ewc.ewc_loss()
                            total_loss = q_loss + ewc_loss
                            
                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()
                    
                    state = next_state
                    
                    if done:
                        break
                
                task_episode_rewards.append(total_reward)
                all_episode_rewards.append(total_reward)
                
                # 타겟 네트워크 업데이트
                if len(task_episode_rewards) % update_frequency == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
                # Epsilon 감소
                epsilon = max(epsilon_end, epsilon * 0.995)
                
                if episode % 10 == 0:
                    avg_reward = np.mean(task_episode_rewards[-10:]) if len(task_episode_rewards) >= 10 else np.mean(task_episode_rewards)
                    print(f"Task {task_id} Episode {episode+1}/{epochs_per_task}, Avg Reward: {avg_reward:.2f}, Eps: {epsilon:.3f}")
            
            # 현재 태스크 완료 후 EWC 파라미터 저장
            ewc.save_optimal_params()
            
            # 태스크별 성능 요약
            task_avg_reward = np.mean(task_episode_rewards)
            print(f"Task {task_id} completed. Average reward: {task_avg_reward:.2f}")
            print(f"Total tasks learned: {continual_buffer.get_task_count()}")
            
            task_id += 1
            
            # 메모리 정리
            if task_id % 2 == 0:
                self.memory_manager.cleanup_memory()
        
        print("Continual Learning DQN training completed!")
        return policy_net, env, continual_buffer
    
    def process_video(self, frames, policy_net=None, ssl_features=None, use_ssl_features=False):
        """비디오 처리 및 결과 생성"""
        print("Processing video...")
        
        if len(frames) == 0:
            print("No frames to process")
            return
        
        # 비디오 작성자 초기화
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, 30.0, (width, height))
        
        # 차선 감지기 초기화
        lane_detector = JetsonOptimizedLaneDetector(self.memory_manager)
        car_x = width // 2
        ssl_dim = int(ssl_features.shape[1]) if (use_ssl_features and ssl_features is not None and hasattr(ssl_features, 'shape')) else 0
        
        # 성능 모니터링
        total_frames = len(frames)
        start_time = time.time()
        
        last_action = 1
        action_names = {0: 'LEFT', 1: 'FORWARD', 2: 'RIGHT'}
        fps = 0.0
        for i, frame in enumerate(frames):
            frame_start_time = time.time()
            
            # 차선 감지
            lanes, lane_state = lane_detector.process_frame(frame)
            
            # 결과 시각화
            overlay = frame.copy()
            output = frame.copy()
            
            if lanes and len(lanes) >= 2:
                left_lane, right_lane = lanes
                
                # 차선 그리기
                if left_lane is not None:
                    cv2.line(overlay, left_lane[0], left_lane[1], (0, 255, 0), 3)
                
                if right_lane is not None:
                    cv2.line(overlay, right_lane[0], right_lane[1], (0, 255, 0), 3)
                
                # 차선 중앙선 그리기
                if left_lane is not None and right_lane is not None:
                    left_center = ((left_lane[0][0] + left_lane[1][0]) // 2,
                                 (left_lane[0][1] + left_lane[1][1]) // 2)
                    right_center = ((right_lane[0][0] + right_lane[1][0]) // 2,
                                   (right_lane[0][1] + right_lane[1][1]) // 2)
                    
                    lane_center = ((left_center[0] + right_center[0]) // 2,
                                  (left_center[1] + right_center[1]) // 2)
                    
                    cv2.circle(overlay, lane_center, 5, (0, 0, 255), -1)
            
            # 정책 기반 액션 추론
            if policy_net is not None:
                # 상태 벡터 구성 (env와 동일 로직)
                state = np.zeros(4, dtype=np.float32)
                if lanes and len(lanes) >= 2:
                    left_lane, right_lane = lanes
                    if left_lane is not None:
                        left_center = (left_lane[0][0] + left_lane[1][0]) / 2
                        state[0] = left_center / width
                    else:
                        state[0] = 0.0
                    if right_lane is not None:
                        right_center = (right_lane[0][0] + right_lane[1][0]) / 2
                        state[1] = right_center / width
                    else:
                        state[1] = 1.0
                    if left_lane is not None and right_lane is not None:
                        lane_center = (left_center + right_center) / 2
                        state[2] = lane_center / width
                    else:
                        state[2] = 0.5
                state[3] = car_x / width
                if use_ssl_features and ssl_features is not None and i < len(ssl_features):
                    feat = ssl_features[i].astype(np.float32)
                    state = np.concatenate([state, feat], axis=0)
                with torch.no_grad():
                    q = policy_net(torch.tensor(state, dtype=torch.float32))
                    action = int(q.argmax().item())
                last_action = action
                # 차량 위치 업데이트 (평가 시뮬레이션)
                if action == 0:
                    car_x = max(0, car_x - 10)
                elif action == 2:
                    car_x = min(width, car_x + 10)
                # 액션 표시
                cv2.putText(overlay, f"Action: {action_names.get(action,'?')}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            # 차선 상태 표시
            cv2.putText(overlay, f"Lane: {lane_state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # FPS 계산
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                fps = self.fps_counter / (time.time() - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # FPS 표시
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 진행률 표시
            progress = (i + 1) / total_frames * 100
            cv2.putText(overlay, f"Progress: {progress:.1f}%", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 프레임 처리 시간 측정
            frame_time = time.time() - frame_start_time
            self.processing_times.append(frame_time)
            
            # 결과 저장
            out.write(overlay)
            
            # 진행률 출력
            if i % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_fps = (i + 1) / elapsed_time
                print(f"Processed {i+1}/{total_frames} frames, "
                      f"Avg FPS: {avg_fps:.1f}, "
                      f"Avg Processing Time: {np.mean(self.processing_times):.3f}s")
        
        out.release()
        
        # 성능 통계 출력
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time
        avg_processing_time = np.mean(self.processing_times)
        
        print(f"\nProcessing completed!")
        print(f"Total frames: {total_frames}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average processing time per frame: {avg_processing_time:.3f}s")
        print(f"Output saved to: {self.output_path}")
    
    def run_continual(self, video_paths, enable_ssl=False, ssl_epochs=5, 
                     epochs_per_task=50, lambda_ewc=1000, replay_ratio=0.3):
        """연속학습 실행 함수"""
        try:
            # 연속학습 DQN 학습
            policy_net, env, continual_buffer = self.train_dqn_continual(
                video_paths=video_paths,
                epochs_per_task=epochs_per_task,
                use_ssl_features=enable_ssl,
                lambda_ewc=lambda_ewc,
                replay_ratio=replay_ratio
            )
            
            # 마지막 태스크로 비디오 처리 (정책 오버레이)
            if len(video_paths) > 0:
                last_frames = self.load_video(video_paths[-1])
                ssl_features = None
                if enable_ssl:
                    ssl_features, _ = self.pretrain_ssl(last_frames, epochs=ssl_epochs)
                
                self.process_video(last_frames, policy_net, ssl_features=ssl_features, use_ssl_features=enable_ssl)
            
            return policy_net, continual_buffer
            
        except Exception as e:
            print(f"Error during continual learning execution: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def run(self, video_path=None):
        """메인 실행 함수 (단일 비디오 또는 디렉토리)"""
        if video_path:
            self.video_path = video_path
        
        if not self.video_path:
            print("Error: No video path provided")
            return
        
        try:
            # 실행 설정: 환경 변수 또는 기본값
            enable_ssl = os.environ.get("ENABLE_SSL", "0") == "1"
            enable_continual = os.environ.get("ENABLE_CONTINUAL", "0") == "1"
            ssl_epochs = int(os.environ.get("SSL_EPOCHS", "5"))
            epochs_per_task = int(os.environ.get("EPOCHS_PER_TASK", "50"))
            lambda_ewc = float(os.environ.get("LAMBDA_EWC", "1000"))
            replay_ratio = float(os.environ.get("REPLAY_RATIO", "0.3"))

            if enable_continual:
                # 연속학습 모드: 여러 비디오 파일을 순차적으로 학습
                if os.path.isdir(self.video_path):
                    # 디렉토리 내 모든 비디오 파일 찾기
                    exts = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
                    video_files = [os.path.join(self.video_path, f) for f in sorted(os.listdir(self.video_path)) 
                                 if os.path.splitext(f)[1] in exts]
                    if len(video_files) == 0:
                        print("No video files found in directory for continual learning")
                        return
                    print(f"Found {len(video_files)} videos for continual learning")
                else:
                    # 단일 파일인 경우
                    video_files = [self.video_path]
                
                self.run_continual(video_files, enable_ssl, ssl_epochs, epochs_per_task, lambda_ewc, replay_ratio)
            else:
                # 기존 단일 비디오 학습 모드
                frames = self.load_video(self.video_path)
                
                ssl_features = None
                if enable_ssl:
                    ssl_features, _ = self.pretrain_ssl(frames, epochs=ssl_epochs)

                # 간단한 DQN 학습 (기존 방식)
                lane_detector = JetsonOptimizedLaneDetector(self.memory_manager)
                env = JetsonDrivingEnv(frames, lane_detector, self.memory_manager,
                                     ssl_features=ssl_features, use_ssl_features=enable_ssl)
                
                # 기본 DQN 학습 (간단한 버전)
                state_dim = 4 + (self.feature_dim if enable_ssl else 0)
                policy_net = JetsonOptimizedDQN(state_dim, 3)
                target_net = JetsonOptimizedDQN(state_dim, 3)
                target_net.load_state_dict(policy_net.state_dict())
                
                optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
                buffer = deque(maxlen=10000)
                
                epochs = int(os.environ.get("DQN_EPOCHS", "50"))
                gamma = 0.99
                epsilon = 0.1
                batch_size = 32
                
                for episode in range(epochs):
                    state, _ = env.reset()
                    total_reward = 0
                    
                    while True:
                        if random.random() < epsilon:
                            action = env.action_space.sample()
                        else:
                            with torch.no_grad():
                                q_values = policy_net(torch.tensor(state, dtype=torch.float32))
                                action = q_values.argmax().item()
                        
                        next_state, reward, done, _, _ = env.step(action)
                        total_reward += reward
                        
                        buffer.append((state, action, reward, next_state, done))
                        
                        if len(buffer) >= batch_size:
                            batch = random.sample(buffer, batch_size)
                            states = torch.tensor([exp[0] for exp in batch], dtype=torch.float32)
                            actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
                            rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
                            next_states = torch.tensor([exp[3] for exp in batch], dtype=torch.float32)
                            dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
                            
                            current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                            with torch.no_grad():
                                next_q_values = target_net(next_states).max(1)[0]
                                target_q_values = rewards + gamma * (1 - dones.float()) * next_q_values
                            
                            loss = nn.MSELoss()(current_q_values, target_q_values)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        
                        state = next_state
                        if done:
                            break
                    
                    if episode % 10 == 0:
                        print(f"Episode {episode}, Reward: {total_reward:.2f}")
                
                # 비디오 처리
                self.process_video(frames, policy_net, ssl_features=ssl_features, use_ssl_features=enable_ssl)
            
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Jetson RL+Continual Learning+SSL Road Segmentation")
    parser.add_argument("video_file", type=str, help="Input video path or directory")
    parser.add_argument("--output", type=str, default="output_jetson.mp4", help="Output video path")
    parser.add_argument("--enable-ssl", action="store_true", help="Enable SSL pretraining + features")
    parser.add_argument("--ssl-epochs", type=int, default=5, help="SSL pretraining epochs")
    parser.add_argument("--feature-dim", type=int, default=128, help="SSL feature dimension")
    parser.add_argument("--enable-continual", action="store_true", help="Enable continual learning")
    parser.add_argument("--epochs-per-task", type=int, default=50, help="Epochs per task in continual learning")
    parser.add_argument("--lambda-ewc", type=float, default=1000, help="EWC regularization strength")
    parser.add_argument("--replay-ratio", type=float, default=0.3, help="Experience replay ratio from previous tasks")
    parser.add_argument("--dqn-epochs", type=int, default=50, help="DQN training episodes (single video mode)")
    args = parser.parse_args()

    print(f"Input video/directory: {args.video_file}")
    print(f"Output video: {args.output}")
    print(f"SSL enabled: {args.enable_ssl}, Continual Learning enabled: {args.enable_continual}")

    # 시스템 초기화
    system = JetsonRoadSegmentationSystem(output_path=args.output,
                                          feature_dim=args.feature_dim,
                                          ssl_input_size=(128, 128))

    # 환경 변수로도 동작하도록 동기화
    os.environ["ENABLE_SSL"] = "1" if args.enable_ssl else "0"
    os.environ["ENABLE_CONTINUAL"] = "1" if args.enable_continual else "0"
    os.environ["DQN_EPOCHS"] = str(args.dqn_epochs)
    os.environ["SSL_EPOCHS"] = str(args.ssl_epochs)
    os.environ["EPOCHS_PER_TASK"] = str(args.epochs_per_task)
    os.environ["LAMBDA_EWC"] = str(args.lambda_ewc)
    os.environ["REPLAY_RATIO"] = str(args.replay_ratio)

    # 실행
    system.run(args.video_file)

if __name__ == "__main__":
    main()
