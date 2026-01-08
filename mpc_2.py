"""
3D MPC 轨迹跟踪与动态避障 - Qt5 图形界面版本
============================================
功能特性：
- Qt5 中文界面，支持实时参数调节
- 可切换显示测量点、预测轨迹等元素
- 卡尔曼滤波演示，带传感器噪声模拟
- 支持 GPU 加速（自动检测）

运行方式：python vtk_3d_avoidance_qt5_cn.py
依赖安装：pip install vtk numpy scipy PyQt5
"""

import sys
import numpy as np
from scipy.optimize import minimize
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QCheckBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QTabWidget, QFrame, QSplitter, QStatusBar, QGridLayout,
    QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import threading
import time
from dataclasses import dataclass, field
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== GPU 检测 ====================
DEVICE = None          # 计算设备
USE_GPU = False        # 是否使用 GPU
TORCH_AVAILABLE = False  # PyTorch 是否可用

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        # NVIDIA GPU
        DEVICE = torch.device('cuda')
        USE_GPU = True
        print(f"[GPU] 检测到 CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU
        DEVICE = torch.device('mps')
        USE_GPU = True
        print("[GPU] 检测到 Apple MPS")
    else:
        DEVICE = torch.device('cpu')
        print("[CPU] 使用 PyTorch CPU 模式")
except ImportError:
    print("[CPU] PyTorch 未安装，使用纯 NumPy 模式")


# ==================== 配置参数类 ====================
@dataclass
class Config:
    """
    配置参数类
    包含所有可调节的仿真参数
    """
    
    # ---------- 目标点参数 ----------
    START_POINT: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # 起点坐标
    END_POINT: np.ndarray = field(default_factory=lambda: np.array([6, 6, 6.0]))   # 终点坐标
    TARGET_SPEED: float = 1.0  # 目标移动速度 (m/s)
    
    # ---------- 障碍物参数 ----------
    OBSTACLE_INITIAL_POS: np.ndarray = field(default_factory=lambda: np.array([5.0, 4.0, 3.0]))  # 初始位置
    OBSTACLE_SPEED_RANGE: Tuple[float, float] = (1.0, 2.0)  # 速度范围 (min, max) m/s
    OBSTACLE_MOMENTUM: float = 0.95      # 动量系数 (0-1)，越大运动越平滑
    OBSTACLE_TURN_RATE: float = 0.02     # 转向触发概率，越小转向越少
    OBSTACLE_TURN_ANGLE_MAX: float = 30  # 最大转向角度 (度)，限制每次转向幅度
    OBSTACLE_RADIUS: float = 0.8         # 障碍物半径 (m)
    SAFETY_MARGIN: float = 0.5           # 安全边距 (m)
    OBSTACLE_BOUNDS: Tuple = ((-1, 6), (-1, 6), (0, 6))  # 运动边界
    
    # ---------- 传感器参数 ----------
    SENSOR_NOISE_STD: float = 0.01     # 位置测量噪声标准差 (m)
    SENSOR_UPDATE_RATE: int = 3       # 传感器更新间隔 (每N步更新一次)
    SENSOR_DROPOUT_PROB: float = 0.15  # 传感器丢帧概率 (0-1)
    
    # ---------- 机器人参数 ----------
    ROBOT_RADIUS: float = 0.35       # 机器人半径 (m)
    MAX_VELOCITY: float = 3.0        # 最大速度 (m/s)
    MAX_ACCELERATION: float = 2.5    # 最大加速度 (m/s²)
    
    # ---------- MPC 参数 ----------
    HORIZON: int = 15          # 预测时域长度
    CONTROL_HORIZON: int = 8   # 控制时域长度
    
    # ---------- 卡尔曼滤波参数 ----------
    KALMAN_PROCESS_NOISE: float = 0.2      # 过程噪声 (运动不确定性)
    KALMAN_MEASUREMENT_NOISE: float = 0.4  # 测量噪声
    
    # ---------- 显示参数 ----------
    ROBOT_TRAIL_LENGTH: int = 40    # 机器人轨迹保留点数
    OBS_TRAIL_LENGTH: int = 40    # 障碍物轨迹保留点数
    PREDICTION_TRAIL_LENGTH: int = 15  # 预测轨迹显示点数
    
    # ---------- 可见性控制 ----------
    SHOW_MEASUREMENTS: bool = True      # 显示测量点（白色）
    SHOW_PREDICTIONS: bool = True       # 显示预测轨迹（黄色）
    SHOW_TRUE_OBSTACLE: bool = True     # 显示真实障碍物（红色半透明）
    SHOW_KALMAN_ESTIMATE: bool = True   # 显示卡尔曼估计（橙色）
    SHOW_SAFETY_BOUNDARY: bool = True   # 显示安全边界


# ==================== 卡尔曼滤波器 ====================
class KalmanFilter3D:
    """
    3D 卡尔曼滤波器
    
    状态向量: [x, y, z, vx, vy, vz, ax, ay, az]
    - 位置 (x, y, z)
    - 速度 (vx, vy, vz)  
    - 加速度 (ax, ay, az)
    
    用途：
    1. 平滑带噪声的传感器测量
    2. 在传感器丢帧时预测障碍物位置
    3. 估计障碍物的速度和加速度
    """
    
    def __init__(self, process_noise=0.3, measurement_noise=0.3):
        """
        初始化卡尔曼滤波器
        
        参数:
            process_noise: 过程噪声，越大表示运动越不可预测
            measurement_noise: 测量噪声，越大表示传感器越不准确
        """
        self.dt = 0.1  # 时间步长 (秒)
        self.n_states = 9  # 状态维度
        
        # 状态向量 [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(self.n_states)
        
        # 状态转移矩阵 (匀加速运动模型)
        # x_new = x + v*dt + 0.5*a*dt²
        # v_new = v + a*dt
        # a_new = a (假设加速度缓慢变化)
        self.F = np.eye(self.n_states)
        self.F[0:3, 3:6] = np.eye(3) * self.dt           # 位置受速度影响
        self.F[0:3, 6:9] = np.eye(3) * 0.5 * self.dt**2  # 位置受加速度影响
        self.F[3:6, 6:9] = np.eye(3) * self.dt           # 速度受加速度影响
        
        # 测量矩阵 (只能测量位置)
        self.H = np.zeros((3, self.n_states))
        self.H[0:3, 0:3] = np.eye(3)
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(self.n_states) * process_noise
        self.Q[6:9, 6:9] *= 2  # 加速度噪声更大
        
        # 测量噪声协方差矩阵
        self.R = np.eye(3) * measurement_noise
        
        # 估计误差协方差矩阵
        self.P = np.eye(self.n_states) * 10
        
        # 状态标志
        self.initialized = False
        self.steps_since_update = 0  # 距离上次测量更新的步数
        
        # 保存噪声参数
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def set_noise(self, process_noise, measurement_noise):
        """动态调整噪声参数"""
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.Q = np.eye(self.n_states) * process_noise
        self.Q[6:9, 6:9] *= 2
        self.R = np.eye(3) * measurement_noise
    
    def predict_only(self):
        """
        仅预测步骤（无测量更新）
        当传感器丢帧时调用
        """
        if self.initialized:
            # 状态预测
            self.x = self.F @ self.x
            # 协方差预测
            self.P = self.F @ self.P @ self.F.T + self.Q
            self.steps_since_update += 1
    
    def update(self, measurement):
        """
        完整的预测+更新步骤
        
        参数:
            measurement: 位置测量值 [x, y, z]
        """
        if not self.initialized:
            # 首次测量，初始化状态
            self.x[0:3] = measurement
            self.initialized = True
            return
        
        # === 预测步骤 ===
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # === 更新步骤 ===
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R  # 新息协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        # 状态更新
        innovation = measurement - self.H @ self.x  # 新息（测量残差）
        self.x = self.x + K @ innovation
        
        # 协方差更新
        self.P = (np.eye(self.n_states) - K @ self.H) @ self.P
        
        self.steps_since_update = 0
    
    def predict_future(self, steps):
        """
        预测未来多步的位置
        
        参数:
            steps: 预测步数
            
        返回:
            predictions: (steps, 3) 数组，每行是一个预测位置
        """
        predictions = []
        x_future = self.x.copy()
        
        for _ in range(steps):
            x_future = self.F @ x_future
            predictions.append(x_future[0:3].copy())
        
        return np.array(predictions)
    
    def get_state(self):
        """获取当前状态估计"""
        return {
            'position': self.x[0:3].copy(),      # 位置估计
            'velocity': self.x[3:6].copy(),      # 速度估计
            'acceleration': self.x[6:9].copy(),  # 加速度估计
            'steps_since_update': self.steps_since_update  # 丢帧计数
        }
    
    def reset(self):
        """重置滤波器状态"""
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 10
        self.initialized = False
        self.steps_since_update = 0


# ==================== 平滑随机运动障碍物 ====================
class SmoothRandomObstacle:
    """
    平滑随机运动的障碍物
    
    特点：
    1. 使用动量系统，运动更平滑
    2. 方向变化有限制，不会突然急转
    3. 速度在范围内随机变化
    4. 碰到边界时反弹
    """
    
    def __init__(self, config: Config):
        """
        初始化障碍物
        
        参数:
            config: 配置参数对象
        """
        self.cfg = config
        self.dt = 0.1  # 时间步长
        self.reset()
    
    def _random_direction(self):
        """生成随机单位方向向量（球面均匀分布）"""
        theta = np.random.uniform(0, 2 * np.pi)  # 方位角
        phi = np.random.uniform(0.2 * np.pi, 0.8 * np.pi)  # 仰角，避免过于垂直
        return np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
    
    def _limit_turn_angle(self, current_dir, target_dir, max_angle_deg):
        """
        限制转向角度
        
        参数:
            current_dir: 当前方向（单位向量）
            target_dir: 目标方向（单位向量）
            max_angle_deg: 最大转向角度（度）
            
        返回:
            new_dir: 限制后的新方向（单位向量）
        """
        # 计算当前方向和目标方向的夹角
        dot = np.clip(np.dot(current_dir, target_dir), -1.0, 1.0)
        angle = np.arccos(dot)  # 弧度
        
        max_angle_rad = np.radians(max_angle_deg)
        
        if angle <= max_angle_rad:
            # 夹角在限制内，直接使用目标方向
            return target_dir
        else:
            # 夹角超过限制，只转向最大角度
            # 使用球面线性插值 (SLERP)
            t = max_angle_rad / angle  # 插值比例
            
            # 简化的插值方法
            new_dir = current_dir + t * (target_dir - current_dir * dot)
            new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-8)
            
            return new_dir
    
    def step(self):
        """
        执行一步运动
        
        返回:
            position: 新位置坐标
        """
        momentum = self.cfg.OBSTACLE_MOMENTUM
        
        # 以一定概率改变目标方向
        if np.random.random() < self.cfg.OBSTACLE_TURN_RATE:
            # 生成新的目标方向
            new_target_dir = self._random_direction()
            
            # 当前方向
            current_speed = np.linalg.norm(self.velocity)
            if current_speed > 0.01:
                current_dir = self.velocity / current_speed
            else:
                current_dir = new_target_dir
            
            # 限制转向角度
            limited_dir = self._limit_turn_angle(
                current_dir, 
                new_target_dir, 
                self.cfg.OBSTACLE_TURN_ANGLE_MAX
            )
            
            # 随机调整速度
            new_speed = np.random.uniform(*self.cfg.OBSTACLE_SPEED_RANGE)
            
            # 设置新的目标速度
            self.target_velocity = limited_dir * new_speed
        
        # 使用动量平滑过渡到目标速度
        # velocity = momentum * velocity + (1 - momentum) * target_velocity
        self.velocity = momentum * self.velocity + (1 - momentum) * self.target_velocity
        
        # 更新位置
        new_pos = self.position + self.velocity * self.dt
        
        # 边界反弹处理
        for i in range(3):
            if new_pos[i] < self.cfg.OBSTACLE_BOUNDS[i][0]:
                new_pos[i] = self.cfg.OBSTACLE_BOUNDS[i][0]
                self.velocity[i] = abs(self.velocity[i])
                self.target_velocity[i] = abs(self.target_velocity[i])
            elif new_pos[i] > self.cfg.OBSTACLE_BOUNDS[i][1]:
                new_pos[i] = self.cfg.OBSTACLE_BOUNDS[i][1]
                self.velocity[i] = -abs(self.velocity[i])
                self.target_velocity[i] = -abs(self.target_velocity[i])
        
        self.position = new_pos
        return self.position.copy()
    
    def reset(self):
        """重置障碍物状态"""
        self.position = self.cfg.OBSTACLE_INITIAL_POS.copy()
        
        # 初始速度
        speed = np.mean(self.cfg.OBSTACLE_SPEED_RANGE)
        direction = self._random_direction()
        self.velocity = direction * speed
        self.target_velocity = self.velocity.copy()


# ==================== 噪声传感器模拟 ====================
class NoisySensor:
    """
    带噪声的传感器模拟
    
    模拟真实传感器的特性：
    1. 测量噪声（高斯噪声）
    2. 有限更新频率
    3. 随机丢帧
    """
    
    def __init__(self, config: Config):
        """初始化传感器"""
        self.cfg = config
        self.step_count = 0
    
    def measure(self, true_position):
        """
        模拟一次测量
        
        参数:
            true_position: 真实位置
            
        返回:
            measurement: 测量值（可能为 None 表示丢帧）
        """
        self.step_count += 1
        
        # 检查是否到达更新周期
        if self.step_count % self.cfg.SENSOR_UPDATE_RATE != 0:
            return None  # 未到更新时间
        
        # 检查是否丢帧
        if np.random.random() < self.cfg.SENSOR_DROPOUT_PROB:
            return None  # 丢帧
        
        # 添加测量噪声
        noise = np.random.normal(0, self.cfg.SENSOR_NOISE_STD, 3)
        measurement = true_position + noise
        
        return measurement
    
    def reset(self):
        """重置传感器"""
        self.step_count = 0


# ==================== MPC 控制器 ====================
class MPCController:
    """
    模型预测控制器 (Model Predictive Control)
    
    功能：
    1. 跟踪目标点
    2. 避开障碍物
    3. 使用卡尔曼滤波预测障碍物轨迹
    """
    
    def __init__(self, config: Config):
        """初始化 MPC 控制器"""
        self.cfg = config
        self.dt = 0.1  # 时间步长
        
        # MPC 参数
        self.horizon = config.HORIZON           # 预测时域
        self.control_horizon = config.CONTROL_HORIZON  # 控制时域
        
        # 约束
        self.max_vel = config.MAX_VELOCITY
        self.max_acc = config.MAX_ACCELERATION
        self.safe_dist = config.OBSTACLE_RADIUS + config.SAFETY_MARGIN
        
        # 代价函数权重
        self.w_goal = 1.0      # 目标跟踪权重
        self.w_control = 0.1   # 控制量权重
        self.w_obs = 3000.0    # 障碍物避让权重
        
        # 状态 [x, y, z, vx, vy, vz]
        self.state = np.concatenate([config.START_POINT.copy(), np.zeros(3)])
        
        # 目标路径参数
        self.start_point = config.START_POINT.copy()
        self.end_point = config.END_POINT.copy()
        self.target_speed = config.TARGET_SPEED
        self.path_vector = self.end_point - self.start_point
        self.path_length = np.linalg.norm(self.path_vector)
        
        # 卡尔曼滤波器
        self.kalman = KalmanFilter3D(
            config.KALMAN_PROCESS_NOISE,
            config.KALMAN_MEASUREMENT_NOISE
        )
        
        # Warm start（热启动，用上一次的解作为初始猜测）
        self.prev_U = None
    
    def update_config(self, config: Config):
        """更新配置参数"""
        self.cfg = config
        self.safe_dist = config.OBSTACLE_RADIUS + config.SAFETY_MARGIN
        self.target_speed = config.TARGET_SPEED
        self.kalman.set_noise(config.KALMAN_PROCESS_NOISE, config.KALMAN_MEASUREMENT_NOISE)
    
    def get_target_position(self, t):
        """
        获取 t 时刻的目标位置（往返运动）
        
        参数:
            t: 时间（秒）
            
        返回:
            position: 目标位置坐标
        """
        total_dist = self.target_speed * t
        cycle_len = 2 * self.path_length  # 一个往返周期的长度
        dist_in_cycle = total_dist % cycle_len
        
        if dist_in_cycle <= self.path_length:
            # 去程
            ratio = dist_in_cycle / self.path_length
            return self.start_point + ratio * self.path_vector
        else:
            # 返程
            ratio = (dist_in_cycle - self.path_length) / self.path_length
            return self.end_point - ratio * self.path_vector
    
    def update_obstacle(self, measurement=None):
        """
        更新障碍物状态估计
        
        参数:
            measurement: 位置测量值（None 表示无测量）
        """
        if measurement is not None:
            self.kalman.update(measurement)
        else:
            self.kalman.predict_only()
    
    def get_obstacle_predictions(self):
        """获取卡尔曼滤波预测的障碍物未来轨迹"""
        return self.kalman.predict_future(self.horizon)
    
    def dynamics(self, state, u):
        """
        机器人动力学模型
        
        参数:
            state: 当前状态 [x, y, z, vx, vy, vz]
            u: 控制输入（加速度）[ax, ay, az]
            
        返回:
            new_state: 下一时刻状态
        """
        pos, vel = state[:3], state[3:]
        
        # 限制加速度
        acc = np.clip(u, -self.max_acc, self.max_acc)
        
        # 更新速度
        vel_new = np.clip(vel + acc * self.dt, -self.max_vel, self.max_vel)
        
        # 更新位置
        pos_new = pos + vel_new * self.dt
        
        return np.concatenate([pos_new, vel_new])
    
    def cost_function(self, U_flat, state, goal_seq, obs_seq):
        """
        MPC 代价函数
        
        参数:
            U_flat: 控制序列（展平）
            state: 当前状态
            goal_seq: 目标位置序列
            obs_seq: 障碍物位置序列（卡尔曼预测）
            
        返回:
            cost: 总代价
        """
        U = U_flat.reshape(self.control_horizon, 3)
        x = state.copy()
        cost = 0.0
        
        for i in range(self.horizon):
            # 获取控制输入
            u = U[i] if i < self.control_horizon else np.zeros(3)
            
            # 状态传播
            x = self.dynamics(x, u)
            
            # 目标跟踪代价
            cost += self.w_goal * np.sum((x[:3] - goal_seq[i])**2)
            
            # 控制量代价
            if i < self.control_horizon:
                cost += self.w_control * np.sum(u**2)
            
            # 障碍物避让代价（软约束）
            dist = np.linalg.norm(x[:3] - obs_seq[i])
            if dist < self.safe_dist:
                cost += self.w_obs * (self.safe_dist - dist)**2
        
        return cost
    
    def compute_control(self, t):
        """
        计算最优控制输入
        
        参数:
            t: 当前时间
            
        返回:
            u: 控制输入 [ax, ay, az]
        """
        # 生成目标位置序列
        goal_seq = np.array([
            self.get_target_position(t + i * self.dt) 
            for i in range(self.horizon)
        ])
        
        # 获取障碍物预测轨迹
        obs_seq = self.get_obstacle_predictions()
        
        # 初始猜测（热启动）
        if self.prev_U is None:
            U0 = np.zeros(self.control_horizon * 3)
        else:
            # 将上一次的解向前滚动
            U0 = np.roll(self.prev_U, -3)
            U0[-3:] = 0
        
        # 优化求解
        bounds = [(-self.max_acc, self.max_acc)] * (self.control_horizon * 3)
        
        result = minimize(
            self.cost_function, U0,
            args=(self.state, goal_seq, obs_seq),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 30, 'disp': False}
        )
        
        self.prev_U = result.x
        return result.x[:3] if result.success else np.zeros(3)
    
    def step(self, t, measurement=None):
        """
        执行一步控制
        
        参数:
            t: 当前时间
            measurement: 障碍物位置测量值
            
        返回:
            position: 机器人新位置
        """
        # 更新障碍物估计
        self.update_obstacle(measurement)
        
        # 计算控制
        u = self.compute_control(t)
        
        # 执行控制
        self.state = self.dynamics(self.state, u)
        
        return self.state[:3].copy()
    
    def reset(self):
        """重置控制器"""
        self.state = np.concatenate([self.cfg.START_POINT.copy(), np.zeros(3)])
        self.prev_U = None
        self.kalman.reset()


# ==================== 仿真器 ====================
class Simulator:
    """
    仿真引擎
    
    在独立线程中运行物理仿真，与 GUI 线程分离
    """
    
    def __init__(self, config: Config):
        """初始化仿真器"""
        self.cfg = config
        self.dt = 0.1
        
        # 创建组件
        self.controller = MPCController(config)
        self.obstacle = SmoothRandomObstacle(config)
        self.sensor = NoisySensor(config)
        
        # 线程同步
        self.lock = threading.Lock()
        self.running = False
        self.paused = True
        self.t = 0.0
        
        # 历史轨迹
        self.robot_history = []
        self.obs_history = []
        self.measurement_history = []
        
        # 当前状态
        self.current_state = self._init_state()
        self.thread = None
    
    def _init_state(self):
        """初始化状态字典"""
        return {
            'robot_pos': self.cfg.START_POINT.copy(),
            'goal_pos': self.cfg.START_POINT.copy(),
            'obs_true_pos': self.cfg.OBSTACLE_INITIAL_POS.copy(),
            'obs_kalman_pos': self.cfg.OBSTACLE_INITIAL_POS.copy(),
            'predicted_obs': np.zeros((self.cfg.HORIZON, 3)),
            'kalman_state': {'velocity': np.zeros(3), 'steps_since_update': 0},
            'time': 0.0,
            'dist_to_obs': 10.0,
            'measurement_available': False,
            'last_measurement': None
        }
    
    def update_config(self, config: Config):
        """更新配置（线程安全）"""
        with self.lock:
            self.cfg = config
            self.controller.update_config(config)
            self.obstacle.cfg = config
            self.sensor.cfg = config
    
    def _loop(self):
        """仿真主循环"""
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            
            start = time.perf_counter()
            
            # 障碍物运动
            obs_true = self.obstacle.step()
            
            # 传感器测量
            measurement = self.sensor.measure(obs_true)
            
            # 控制器步进
            robot_pos = self.controller.step(self.t, measurement)
            goal_pos = self.controller.get_target_position(self.t)
            
            # 获取卡尔曼滤波状态
            kalman_state = self.controller.kalman.get_state()
            obs_kalman = kalman_state['position']
            predicted = self.controller.get_obstacle_predictions()
            
            # 计算距离
            dist = np.linalg.norm(robot_pos - obs_true)
            
            # 更新状态（线程安全）
            with self.lock:
                self.current_state = {
                    'robot_pos': robot_pos,
                    'goal_pos': goal_pos,
                    'obs_true_pos': obs_true,
                    'obs_kalman_pos': obs_kalman,
                    'predicted_obs': predicted,
                    'kalman_state': kalman_state,
                    'time': self.t,
                    'dist_to_obs': dist,
                    'measurement_available': measurement is not None,
                    'last_measurement': measurement
                }
                
                # 更新历史
                self.robot_history.append(robot_pos.copy())
                self.obs_history.append(obs_true.copy())
                if measurement is not None:
                    self.measurement_history.append(measurement.copy())
                
                # 限制历史长度
                max_len = self.cfg.ROBOT_TRAIL_LENGTH
                if len(self.robot_history) > max_len:
                    self.robot_history = self.robot_history[-max_len:]
                if len(self.obs_history) > self.cfg.OBS_TRAIL_LENGTH:
                    self.obs_history = self.obs_history[-self.cfg.OBS_TRAIL_LENGTH:]
                if len(self.measurement_history) > 50:
                    self.measurement_history = self.measurement_history[-50:]
            
            self.t += self.dt
            
            # 控制帧率
            elapsed = time.perf_counter() - start
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
    
    def start(self):
        """启动仿真线程"""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止仿真"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def pause(self):
        """暂停仿真"""
        self.paused = True
    
    def resume(self):
        """恢复仿真"""
        self.paused = False
    
    def reset(self):
        """重置仿真"""
        self.paused = True
        time.sleep(0.05)
        
        self.t = 0.0
        self.controller.reset()
        self.obstacle.reset()
        self.sensor.reset()
        
        with self.lock:
            self.robot_history.clear()
            self.obs_history.clear()
            self.measurement_history.clear()
            self.current_state = self._init_state()
    
    def get_state(self):
        """获取当前状态（线程安全）"""
        with self.lock:
            return (
                self.current_state.copy(),
                list(self.robot_history),
                list(self.obs_history),
                list(self.measurement_history)
            )


# ==================== VTK 渲染组件 ====================
class VTKWidget(QVTKRenderWindowInteractor):
    """
    VTK 3D 渲染组件
    
    负责 3D 场景的显示和更新
    """
    
    def __init__(self, parent, config: Config):
        """初始化 VTK 组件"""
        super().__init__(parent)
        self.cfg = config
        
        # 创建渲染器
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        
        # 设置交互风格
        self.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # 初始化场景
        self._setup_scene()
    
    def _create_sphere(self, radius, color, opacity=1.0):
        """
        创建球体 Actor
        
        参数:
            radius: 半径
            color: RGB 颜色 (0-1, 0-1, 0-1)
            opacity: 不透明度 (0-1)
        """
        source = vtk.vtkSphereSource()
        source.SetRadius(radius)
        source.SetThetaResolution(24)
        source.SetPhiResolution(24)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.3)
        
        return actor
    
    def _create_trail(self, color, radius=0.04, opacity=1.0):
        """创建轨迹管道"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(8)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        
        return actor, points, lines, polydata, tube
    
    def _setup_scene(self):
        """初始化 3D 场景"""
        # 背景渐变
        self.renderer.SetBackground(0.05, 0.05, 0.1)    # 底部颜色
        self.renderer.SetBackground2(0.15, 0.15, 0.25)  # 顶部颜色
        self.renderer.GradientBackgroundOn()
        
        # 起点标记（绿色）
        start_m = self._create_sphere(0.2, (0.2, 0.9, 0.2), 0.8)
        start_m.SetPosition(self.cfg.START_POINT)
        self.renderer.AddActor(start_m)
        
        # 终点标记（紫色）
        end_m = self._create_sphere(0.2, (0.9, 0.2, 0.9), 0.8)
        end_m.SetPosition(self.cfg.END_POINT)
        self.renderer.AddActor(end_m)
        
        # 路径线
        self._create_path_line()
        
        # 机器人（蓝色）
        self.robot_actor = self._create_sphere(self.cfg.ROBOT_RADIUS, (0.2, 0.6, 1.0))
        self.renderer.AddActor(self.robot_actor)
        
        # 障碍物真实位置（红色半透明）
        self.obs_true_actor = self._create_sphere(self.cfg.OBSTACLE_RADIUS, (1.0, 0.2, 0.2), 0.4)
        self.renderer.AddActor(self.obs_true_actor)
        
        # 障碍物卡尔曼估计（橙色）
        self.obs_kalman_actor = self._create_sphere(self.cfg.OBSTACLE_RADIUS * 0.8, (1.0, 0.6, 0.0), 0.9)
        self.renderer.AddActor(self.obs_kalman_actor)
        
        # 安全边界
        safe_r = self.cfg.OBSTACLE_RADIUS + self.cfg.SAFETY_MARGIN
        self.safety_actor = self._create_sphere(safe_r, (1.0, 0.5, 0.5), 0.15)
        self.renderer.AddActor(self.safety_actor)
        
        # 目标点（绿色）
        self.goal_actor = self._create_sphere(0.2, (0.3, 1.0, 0.4))
        self.renderer.AddActor(self.goal_actor)
        
        # 测量点（白色小球）
        self.measurement_actors = []
        for _ in range(50):
            actor = self._create_sphere(0.1, (1.0, 1.0, 1.0), 0.7)
            actor.VisibilityOff()
            self.renderer.AddActor(actor)
            self.measurement_actors.append(actor)
        
        # 预测轨迹（黄色小球）
        self.prediction_actors = []
        for _ in range(self.cfg.PREDICTION_TRAIL_LENGTH):
            actor = self._create_sphere(0.08, (1.0, 0.9, 0.2), 0.6)
            actor.VisibilityOff()
            self.renderer.AddActor(actor)
            self.prediction_actors.append(actor)
        
        # 机器人轨迹（蓝色管道）
        (self.robot_trail_actor, self.robot_trail_points,
         self.robot_trail_lines, self.robot_trail_polydata,
         self.robot_trail_tube) = self._create_trail((0.3, 0.7, 1.0), 0.05)
        self.renderer.AddActor(self.robot_trail_actor)
        
        # 障碍物轨迹（红色管道）
        (self.obs_trail_actor, self.obs_trail_points,
         self.obs_trail_lines, self.obs_trail_polydata,
         self.obs_trail_tube) = self._create_trail((1.0, 0.4, 0.4), 0.02, 0.5)
        self.renderer.AddActor(self.obs_trail_actor)
        
        # 地面网格
        self._create_ground()
        
        # 坐标轴
        self._create_axes()
        
        # 光源
        for pos, intensity in [((15, 15, 20), 0.8), ((-10, -10, 15), 0.4)]:
            light = vtk.vtkLight()
            light.SetPosition(*pos)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)
        
        # 相机
        center = (self.cfg.START_POINT + self.cfg.END_POINT) / 2
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0] + 18, center[1] + 18, center[2] + 14)
        camera.SetFocalPoint(center)
        camera.SetViewUp(0, 0, 1)
    
    def _create_path_line(self):
        """创建目标路径线"""
        points = vtk.vtkPoints()
        points.InsertNextPoint(self.cfg.START_POINT)
        points.InsertNextPoint(self.cfg.END_POINT)
        
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 1.0, 0.5)
        actor.GetProperty().SetLineWidth(2)
        actor.GetProperty().SetOpacity(0.5)
        self.renderer.AddActor(actor)
    
    def _create_ground(self):
        """创建地面网格"""
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-2, -2, -0.05)
        plane.SetPoint1(14, -2, -0.05)
        plane.SetPoint2(-2, 12, -0.05)
        plane.SetXResolution(16)
        plane.SetYResolution(14)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.3, 0.3, 0.35)
        actor.GetProperty().SetOpacity(0.3)
        self.renderer.AddActor(actor)
    
    def _create_axes(self):
        """创建坐标轴（无文字）"""
        for color, end in [((1, 0, 0), (2, 0, 0)),    # X 红色
                           ((0, 1, 0), (0, 2, 0)),    # Y 绿色
                           ((0, 0, 1), (0, 0, 2))]:   # Z 蓝色
            line = vtk.vtkLineSource()
            line.SetPoint1(0, 0, 0)
            line.SetPoint2(*end)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetLineWidth(2)
            self.renderer.AddActor(actor)
    
    def _update_trail(self, points, lines, polydata, history):
        """更新轨迹显示"""
        points.Reset()
        lines.Reset()
        
        if len(history) < 2:
            polydata.Modified()
            return
        
        for p in history:
            points.InsertNextPoint(p)
        
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(history))
        for i in range(len(history)):
            line.GetPointIds().SetId(i, i)
        lines.InsertNextCell(line)
        
        polydata.Modified()
    
    def update_display(self, state, robot_hist, obs_hist, meas_hist, config):
        """
        更新 3D 显示
        
        参数:
            state: 当前状态字典
            robot_hist: 机器人轨迹历史
            obs_hist: 障碍物轨迹历史
            meas_hist: 测量点历史
            config: 配置参数
        """
        self.cfg = config
        
        # 更新位置
        self.robot_actor.SetPosition(state['robot_pos'])
        self.goal_actor.SetPosition(state['goal_pos'])
        self.obs_true_actor.SetPosition(state['obs_true_pos'])
        self.obs_kalman_actor.SetPosition(state['obs_kalman_pos'])
        self.safety_actor.SetPosition(state['obs_kalman_pos'])
        
        # 更新可见性
        self.obs_true_actor.SetVisibility(config.SHOW_TRUE_OBSTACLE)
        self.obs_kalman_actor.SetVisibility(config.SHOW_KALMAN_ESTIMATE)
        self.safety_actor.SetVisibility(config.SHOW_SAFETY_BOUNDARY)
        
        # 更新轨迹
        self._update_trail(self.robot_trail_points, self.robot_trail_lines,
                          self.robot_trail_polydata, robot_hist)
        self.robot_trail_tube.Modified()
        
        self._update_trail(self.obs_trail_points, self.obs_trail_lines,
                          self.obs_trail_polydata, obs_hist)
        self.obs_trail_tube.Modified()
        
        # 更新测量点显示
        for i, actor in enumerate(self.measurement_actors):
            if config.SHOW_MEASUREMENTS and i < len(meas_hist):
                actor.SetPosition(meas_hist[i])
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        
        # 更新预测轨迹显示
        for i, actor in enumerate(self.prediction_actors):
            if config.SHOW_PREDICTIONS and i < len(state['predicted_obs']):
                actor.SetPosition(state['predicted_obs'][i])
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        
        # 根据距离更新机器人颜色
        safe_dist = config.OBSTACLE_RADIUS + config.SAFETY_MARGIN
        if state['dist_to_obs'] < safe_dist * 1.3:
            danger = min(1, (safe_dist * 1.3 - state['dist_to_obs']) / (safe_dist * 0.3))
            self.robot_actor.GetProperty().SetColor(
                0.2 + 0.8 * danger,      # R: 蓝→红
                0.6 * (1 - danger),      # G: 降低
                1.0 * (1 - danger)       # B: 降低
            )
        else:
            self.robot_actor.GetProperty().SetColor(0.2, 0.6, 1.0)
        
        # 刷新显示
        self.GetRenderWindow().Render()


# ==================== 控制面板 ====================
class ControlPanel(QWidget):
    """
    控制面板
    
    包含所有可调参数的 GUI 控件
    """
    
    def __init__(self, config: Config, parent=None):
        """初始化控制面板"""
        super().__init__(parent)
        self.cfg = config
        self._setup_ui()
    
    def _setup_ui(self):
        """构建用户界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("控制面板")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 选项卡
        tabs = QTabWidget()
        
        # ========== 传感器选项卡 ==========
        sensor_tab = QWidget()
        sensor_layout = QVBoxLayout(sensor_tab)
        
        sensor_group = QGroupBox("传感器参数")
        sensor_grid = QGridLayout(sensor_group)
        
        # 噪声标准差
        sensor_grid.addWidget(QLabel("噪声标准差 (m):"), 0, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 2.0)
        self.noise_spin.setSingleStep(0.1)
        self.noise_spin.setValue(self.cfg.SENSOR_NOISE_STD)
        self.noise_spin.setToolTip("传感器测量噪声的标准差，越大测量越不准确")
        self.noise_spin.valueChanged.connect(self._on_sensor_changed)
        sensor_grid.addWidget(self.noise_spin, 0, 1)
        
        # 更新间隔
        sensor_grid.addWidget(QLabel("更新间隔 (步):"), 1, 0)
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(1, 10)
        self.update_rate_spin.setValue(self.cfg.SENSOR_UPDATE_RATE)
        self.update_rate_spin.setToolTip("每隔多少步更新一次传感器读数")
        self.update_rate_spin.valueChanged.connect(self._on_sensor_changed)
        sensor_grid.addWidget(self.update_rate_spin, 1, 1)
        
        # 丢帧概率
        sensor_grid.addWidget(QLabel("丢帧概率 (%):"), 2, 0)
        self.dropout_spin = QSpinBox()
        self.dropout_spin.setRange(0, 50)
        self.dropout_spin.setValue(int(self.cfg.SENSOR_DROPOUT_PROB * 100))
        self.dropout_spin.setToolTip("传感器数据丢失的概率")
        self.dropout_spin.valueChanged.connect(self._on_sensor_changed)
        sensor_grid.addWidget(self.dropout_spin, 2, 1)
        
        sensor_layout.addWidget(sensor_group)
        sensor_layout.addStretch()
        tabs.addTab(sensor_tab, "传感器")
        
        # ========== 卡尔曼选项卡 ==========
        kalman_tab = QWidget()
        kalman_layout = QVBoxLayout(kalman_tab)
        
        kalman_group = QGroupBox("卡尔曼滤波参数")
        kalman_grid = QGridLayout(kalman_group)
        
        # 过程噪声
        kalman_grid.addWidget(QLabel("过程噪声:"), 0, 0)
        self.process_noise_spin = QDoubleSpinBox()
        self.process_noise_spin.setRange(0.01, 2.0)
        self.process_noise_spin.setSingleStep(0.1)
        self.process_noise_spin.setValue(self.cfg.KALMAN_PROCESS_NOISE)
        self.process_noise_spin.setToolTip("运动不确定性，越大表示障碍物运动越不可预测")
        self.process_noise_spin.valueChanged.connect(self._on_kalman_changed)
        kalman_grid.addWidget(self.process_noise_spin, 0, 1)
        
        # 测量噪声
        kalman_grid.addWidget(QLabel("测量噪声:"), 1, 0)
        self.meas_noise_spin = QDoubleSpinBox()
        self.meas_noise_spin.setRange(0.01, 2.0)
        self.meas_noise_spin.setSingleStep(0.1)
        self.meas_noise_spin.setValue(self.cfg.KALMAN_MEASUREMENT_NOISE)
        self.meas_noise_spin.setToolTip("测量不确定性，越大表示越不信任测量值")
        self.meas_noise_spin.valueChanged.connect(self._on_kalman_changed)
        kalman_grid.addWidget(self.meas_noise_spin, 1, 1)
        
        kalman_layout.addWidget(kalman_group)
        kalman_layout.addStretch()
        tabs.addTab(kalman_tab, "卡尔曼")
        
        # ========== 障碍物选项卡 ==========
        obs_tab = QWidget()
        obs_layout = QVBoxLayout(obs_tab)
        
        # 运动参数组
        motion_group = QGroupBox("运动参数")
        motion_grid = QGridLayout(motion_group)
        
        # 动量
        motion_grid.addWidget(QLabel("动量系数:"), 0, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.5, 0.99)
        self.momentum_spin.setSingleStep(0.05)
        self.momentum_spin.setValue(self.cfg.OBSTACLE_MOMENTUM)
        self.momentum_spin.setToolTip("越大运动越平滑，方向变化越慢")
        self.momentum_spin.valueChanged.connect(self._on_obstacle_changed)
        motion_grid.addWidget(self.momentum_spin, 0, 1)
        
        # 转向概率
        motion_grid.addWidget(QLabel("转向概率:"), 1, 0)
        self.turn_rate_spin = QDoubleSpinBox()
        self.turn_rate_spin.setRange(0.005, 0.2)
        self.turn_rate_spin.setSingleStep(0.005)
        self.turn_rate_spin.setDecimals(3)
        self.turn_rate_spin.setValue(self.cfg.OBSTACLE_TURN_RATE)
        self.turn_rate_spin.setToolTip("每一步改变方向的概率，越小转向越少")
        self.turn_rate_spin.valueChanged.connect(self._on_obstacle_changed)
        motion_grid.addWidget(self.turn_rate_spin, 1, 1)
        
        # 最大转向角度
        motion_grid.addWidget(QLabel("最大转角 (度):"), 2, 0)
        self.turn_angle_spin = QDoubleSpinBox()
        self.turn_angle_spin.setRange(5, 180)
        self.turn_angle_spin.setSingleStep(5)
        self.turn_angle_spin.setValue(self.cfg.OBSTACLE_TURN_ANGLE_MAX)
        self.turn_angle_spin.setToolTip("每次转向的最大角度，越小转弯越平缓")
        self.turn_angle_spin.valueChanged.connect(self._on_obstacle_changed)
        motion_grid.addWidget(self.turn_angle_spin, 2, 1)
        
        obs_layout.addWidget(motion_group)
        
        # 速度参数组
        speed_group = QGroupBox("速度参数")
        speed_grid = QGridLayout(speed_group)
        
        speed_grid.addWidget(QLabel("最小速度 (m/s):"), 0, 0)
        self.speed_min_spin = QDoubleSpinBox()
        self.speed_min_spin.setRange(0.1, 3.0)
        self.speed_min_spin.setSingleStep(0.1)
        self.speed_min_spin.setValue(self.cfg.OBSTACLE_SPEED_RANGE[0])
        self.speed_min_spin.valueChanged.connect(self._on_obstacle_changed)
        speed_grid.addWidget(self.speed_min_spin, 0, 1)
        
        speed_grid.addWidget(QLabel("最大速度 (m/s):"), 1, 0)
        self.speed_max_spin = QDoubleSpinBox()
        self.speed_max_spin.setRange(0.5, 5.0)
        self.speed_max_spin.setSingleStep(0.1)
        self.speed_max_spin.setValue(self.cfg.OBSTACLE_SPEED_RANGE[1])
        self.speed_max_spin.valueChanged.connect(self._on_obstacle_changed)
        speed_grid.addWidget(self.speed_max_spin, 1, 1)
        
        obs_layout.addWidget(speed_group)
        obs_layout.addStretch()
        tabs.addTab(obs_tab, "障碍物")
        
        # ========== 显示选项卡 ==========
        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        
        # 可见性设置
        vis_group = QGroupBox("可见性设置")
        vis_layout = QVBoxLayout(vis_group)
        
        self.show_meas_cb = QCheckBox("显示测量点（白色）")
        self.show_meas_cb.setChecked(self.cfg.SHOW_MEASUREMENTS)
        self.show_meas_cb.setToolTip("显示带噪声的传感器测量位置")
        self.show_meas_cb.stateChanged.connect(self._on_display_changed)
        vis_layout.addWidget(self.show_meas_cb)
        
        self.show_pred_cb = QCheckBox("显示预测轨迹（黄色）")
        self.show_pred_cb.setChecked(self.cfg.SHOW_PREDICTIONS)
        self.show_pred_cb.setToolTip("显示卡尔曼滤波预测的障碍物未来位置")
        self.show_pred_cb.stateChanged.connect(self._on_display_changed)
        vis_layout.addWidget(self.show_pred_cb)
        
        self.show_true_cb = QCheckBox("显示真实障碍物（红色）")
        self.show_true_cb.setChecked(self.cfg.SHOW_TRUE_OBSTACLE)
        self.show_true_cb.setToolTip("显示障碍物的真实位置（半透明）")
        self.show_true_cb.stateChanged.connect(self._on_display_changed)
        vis_layout.addWidget(self.show_true_cb)
        
        self.show_kalman_cb = QCheckBox("显示卡尔曼估计（橙色）")
        self.show_kalman_cb.setChecked(self.cfg.SHOW_KALMAN_ESTIMATE)
        self.show_kalman_cb.setToolTip("显示卡尔曼滤波估计的障碍物位置")
        self.show_kalman_cb.stateChanged.connect(self._on_display_changed)
        vis_layout.addWidget(self.show_kalman_cb)
        
        self.show_safety_cb = QCheckBox("显示安全边界")
        self.show_safety_cb.setChecked(self.cfg.SHOW_SAFETY_BOUNDARY)
        self.show_safety_cb.setToolTip("显示机器人需要避开的安全区域")
        self.show_safety_cb.stateChanged.connect(self._on_display_changed)
        vis_layout.addWidget(self.show_safety_cb)
        
        display_layout.addWidget(vis_group)
        
        # 目标速度
        target_group = QGroupBox("目标参数")
        target_grid = QGridLayout(target_group)
        
        target_grid.addWidget(QLabel("移动速度 (m/s):"), 0, 0)
        self.target_speed_spin = QDoubleSpinBox()
        self.target_speed_spin.setRange(0.1, 3.0)
        self.target_speed_spin.setSingleStep(0.1)
        self.target_speed_spin.setValue(self.cfg.TARGET_SPEED)
        self.target_speed_spin.setToolTip("目标点的移动速度")
        self.target_speed_spin.valueChanged.connect(self._on_target_changed)
        target_grid.addWidget(self.target_speed_spin, 0, 1)
        
        display_layout.addWidget(target_group)
        display_layout.addStretch()
        tabs.addTab(display_tab, "显示")
        
        layout.addWidget(tabs)
        
        # ========== 控制按钮 ==========
        btn_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("开始")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self.play_btn)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)
        
        layout.addLayout(btn_layout)
        
        # ========== 状态显示 ==========
        info_group = QGroupBox("状态信息")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("时间: 0.0秒")
        self.info_label.setFont(QFont("Courier New", 10))
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(info_group)
        
        # 回调函数
        self.on_config_changed = None
        self.on_play = None
        self.on_reset = None
    
    def _on_sensor_changed(self):
        """传感器参数改变"""
        self.cfg.SENSOR_NOISE_STD = self.noise_spin.value()
        self.cfg.SENSOR_UPDATE_RATE = self.update_rate_spin.value()
        self.cfg.SENSOR_DROPOUT_PROB = self.dropout_spin.value() / 100.0
        if self.on_config_changed:
            self.on_config_changed(self.cfg)
    
    def _on_kalman_changed(self):
        """卡尔曼参数改变"""
        self.cfg.KALMAN_PROCESS_NOISE = self.process_noise_spin.value()
        self.cfg.KALMAN_MEASUREMENT_NOISE = self.meas_noise_spin.value()
        if self.on_config_changed:
            self.on_config_changed(self.cfg)
    
    def _on_obstacle_changed(self):
        """障碍物参数改变"""
        self.cfg.OBSTACLE_MOMENTUM = self.momentum_spin.value()
        self.cfg.OBSTACLE_TURN_RATE = self.turn_rate_spin.value()
        self.cfg.OBSTACLE_TURN_ANGLE_MAX = self.turn_angle_spin.value()
        self.cfg.OBSTACLE_SPEED_RANGE = (
            self.speed_min_spin.value(),
            self.speed_max_spin.value()
        )
        if self.on_config_changed:
            self.on_config_changed(self.cfg)
    
    def _on_display_changed(self):
        """显示设置改变"""
        self.cfg.SHOW_MEASUREMENTS = self.show_meas_cb.isChecked()
        self.cfg.SHOW_PREDICTIONS = self.show_pred_cb.isChecked()
        self.cfg.SHOW_TRUE_OBSTACLE = self.show_true_cb.isChecked()
        self.cfg.SHOW_KALMAN_ESTIMATE = self.show_kalman_cb.isChecked()
        self.cfg.SHOW_SAFETY_BOUNDARY = self.show_safety_cb.isChecked()
        if self.on_config_changed:
            self.on_config_changed(self.cfg)
    
    def _on_target_changed(self):
        """目标参数改变"""
        self.cfg.TARGET_SPEED = self.target_speed_spin.value()
        if self.on_config_changed:
            self.on_config_changed(self.cfg)
    
    def _on_play(self, checked):
        """开始/暂停按钮"""
        self.play_btn.setText("暂停" if checked else "开始")
        if self.on_play:
            self.on_play(checked)
    
    def _on_reset(self):
        """重置按钮"""
        self.play_btn.setChecked(False)
        self.play_btn.setText("开始")
        if self.on_reset:
            self.on_reset()
    
    def update_info(self, state):
        """更新状态信息显示"""
        ks = state['kalman_state']
        vel = ks['velocity']
        speed = np.linalg.norm(vel)
        est_error = np.linalg.norm(state['obs_kalman_pos'] - state['obs_true_pos'])
        safe_dist = self.cfg.OBSTACLE_RADIUS + self.cfg.SAFETY_MARGIN
        
        status = "避障中" if state['dist_to_obs'] < safe_dist else "安全"
        meas = "有" if state['measurement_available'] else f"无 ({ks['steps_since_update']}步)"
        
        text = (
            f"时间: {state['time']:.1f}秒\n"
            f"状态: {status}\n"
            f"障碍物距离: {state['dist_to_obs']:.2f}m\n"
            f"估计误差: {est_error:.3f}m\n"
            f"障碍物速度: {speed:.2f}m/s\n"
            f"本步测量: {meas}"
        )
        self.info_label.setText(text)


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    """应用程序主窗口"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        
        # 创建配置和仿真器
        self.cfg = Config()
        self.simulator = Simulator(self.cfg)
        
        # 构建界面
        self._setup_ui()
        
        # 启动定时器
        self._setup_timer()
        
        # 启动仿真线程
        self.simulator.start()
    
    def _setup_ui(self):
        """构建用户界面"""
        self.setWindowTitle("3D MPC 动态避障仿真 - 卡尔曼滤波演示")
        self.setGeometry(100, 100, 1600, 900)
        
        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        
        # 主布局
        main_layout = QHBoxLayout(central)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # VTK 3D 视图
        self.vtk_widget = VTKWidget(splitter, self.cfg)
        
        # 控制面板
        self.control_panel = ControlPanel(self.cfg, splitter)
        self.control_panel.setMaximumWidth(320)
        self.control_panel.setMinimumWidth(280)
        
        # 连接回调
        self.control_panel.on_config_changed = self._on_config_changed
        self.control_panel.on_play = self._on_play
        self.control_panel.on_reset = self._on_reset
        
        splitter.addWidget(self.vtk_widget)
        splitter.addWidget(self.control_panel)
        splitter.setSizes([1280, 320])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage("就绪 - 点击 [开始] 运行仿真")
    
    def _setup_timer(self):
        """设置刷新定时器"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(33)  # 约 30 FPS
    
    def _on_config_changed(self, config):
        """配置改变回调"""
        self.cfg = config
        self.simulator.update_config(config)
    
    def _on_play(self, play):
        """开始/暂停回调"""
        if play:
            self.simulator.resume()
            self.statusBar().showMessage("运行中...")
        else:
            self.simulator.pause()
            self.statusBar().showMessage("已暂停")
    
    def _on_reset(self):
        """重置回调"""
        self.simulator.reset()
        self.statusBar().showMessage("已重置 - 点击 [开始] 运行仿真")
    
    def _update(self):
        """定时刷新"""
        state, robot_hist, obs_hist, meas_hist = self.simulator.get_state()
        self.vtk_widget.update_display(state, robot_hist, obs_hist, meas_hist, self.cfg)
        self.control_panel.update_info(state)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.simulator.stop()
        self.vtk_widget.close()
        event.accept()


# ==================== 主程序入口 ====================
def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用风格
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 初始化 VTK 交互器
    window.vtk_widget.Initialize()
    window.vtk_widget.Start()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()# """
