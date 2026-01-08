
"""
3D MPC Trajectory Tracking with Obstacle Avoidance - GPU Accelerated Version
==============================================================================
Features:
1. GPU acceleration with PyTorch (auto-detect CUDA/MPS/CPU)
2. Multi-threaded rendering and computation
3. Kalman filter for obstacle trajectory prediction
4. Random obstacle motion with configurable speed range

Run: python vtk_3d_avoidance_gpu.py
Dependencies: pip install vtk numpy scipy torch
# Copyright (c) 2025, Chen XingYu. All rights reserved.
#
# License: Non-Commercial Use Only / 仅限非商业使用
# -----------------------------------------------------------------------------
# 本代码及其衍生作品仅允许用于个人学习、学术研究与教学等非商业场景。
# 严禁任何形式的商业使用，包括但不限于：出售、付费服务、SaaS/在线服务、
# 广告变现、集成到商业产品或用于商业咨询/竞赛/投标等。如需商业授权，请
# 先行获得版权所有者书面许可并签署授权协议。
#
# 允许的非商业使用条件：
# 1) 保留本版权与许可声明；2) 在衍生作品/发表物中进行署名（Lu Yaoheng）并
# 标明来源仓库；3) 不得移除或修改本段声明。
#
# 免责声明：本代码按“现状”提供，不含任何明示或默示担保。作者不对因使用本
# 代码产生的任何直接或间接损失承担责任。使用者需自行评估并承担风险。
#
# English Summary:
# This code is provided for personal, academic, and research purposes only.
# Any commercial use (sale, paid service, SaaS, ad-monetization, integration
# into commercial products, consultancy, competitions, bids, etc.) is strictly
# prohibited without prior written permission from the copyright holder.
# Keep this notice intact and provide proper attribution in derived works.
# Provided "as is" without warranty of any kind. Use at your own risk.
#
# Contact / 商务与授权联系: <cdssywc@163.com>
"""

import numpy as np
from scipy.optimize import minimize
import vtk
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============== GPU/PyTorch Detection ==============
DEVICE = None
USE_GPU = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        USE_GPU = True
        print(f"[GPU] CUDA detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        USE_GPU = True
        print("[GPU] Apple MPS detected")
    else:
        DEVICE = torch.device('cpu')
        print("[CPU] No GPU detected, using CPU with PyTorch optimization")
except ImportError:
    print("[CPU] PyTorch not installed, using NumPy only")


# ============== Configuration ==============
@dataclass
class Config:
    """Configuration parameters"""
    
    # Target motion
    START_POINT: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    END_POINT: np.ndarray = field(default_factory=lambda: np.array([10.0, 8.0, 6.0]))
    TARGET_SPEED: float = 1.2
    
    # Obstacle parameters
    OBSTACLE_INITIAL_POS: np.ndarray = field(default_factory=lambda: np.array([5.0, 4.0, 3.0]))
    OBSTACLE_SPEED_RANGE: Tuple[float, float] = (0.8, 2.5)
    OBSTACLE_SPEED_CHANGE_RATE: float = 0.5
    OBSTACLE_DIRECTION_CHANGE_RATE: float = 0.15
    OBSTACLE_RADIUS: float = 0.8
    SAFETY_MARGIN: float = 0.5
    OBSTACLE_BOUNDS: Tuple = ((-1, 6), (-1, 6), (0, 8))
    
    # Robot parameters
    ROBOT_RADIUS: float = 0.35
    MAX_VELOCITY: float = 3.0
    MAX_ACCELERATION: float = 2.5
    
    # MPC parameters
    HORIZON: int = 15
    CONTROL_HORIZON: int = 8
    MPC_ITERATIONS: int = 30
    
    # Kalman filter
    KALMAN_PROCESS_NOISE: float = 0.8
    KALMAN_MEASUREMENT_NOISE: float = 0.1
    
    # Trail lengths
    ROBOT_TRAIL_LENGTH: int = 600
    GOAL_TRAIL_LENGTH: int = 600
    OBS_TRAIL_LENGTH: int = 400
    PREDICTION_TRAIL_LENGTH: int = 15
    
    # Display
    WINDOW_SIZE: Tuple[int, int] = (1500, 950)


cfg = Config()


# ============== GPU-Accelerated Kalman Filter ==============
class KalmanFilter3D:
    """
    3D Kalman Filter for obstacle motion prediction
    State: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    
    def __init__(self, process_noise=0.5, measurement_noise=0.1):
        self.dt = 0.1
        self.n_states = 9
        self.n_meas = 3
        
        if TORCH_AVAILABLE:
            self._init_torch(process_noise, measurement_noise)
        else:
            self._init_numpy(process_noise, measurement_noise)
    
    def _init_torch(self, pn, mn):
        """Initialize with PyTorch tensors"""
        self.use_torch = True
        dtype = torch.float32
        
        self.x = torch.zeros(self.n_states, dtype=dtype, device=DEVICE)
        
        # State transition matrix
        self.F = torch.eye(self.n_states, dtype=dtype, device=DEVICE)
        self.F[0:3, 3:6] = torch.eye(3, dtype=dtype, device=DEVICE) * self.dt
        self.F[0:3, 6:9] = torch.eye(3, dtype=dtype, device=DEVICE) * 0.5 * self.dt**2
        self.F[3:6, 6:9] = torch.eye(3, dtype=dtype, device=DEVICE) * self.dt
        
        # Measurement matrix
        self.H = torch.zeros(self.n_meas, self.n_states, dtype=dtype, device=DEVICE)
        self.H[0:3, 0:3] = torch.eye(3, dtype=dtype, device=DEVICE)
        
        # Noise covariances
        self.Q = torch.eye(self.n_states, dtype=dtype, device=DEVICE) * pn
        self.Q[6:9, 6:9] *= 2
        self.R = torch.eye(self.n_meas, dtype=dtype, device=DEVICE) * mn
        
        # Error covariance
        self.P = torch.eye(self.n_states, dtype=dtype, device=DEVICE)
        
        self.initialized = False
    
    def _init_numpy(self, pn, mn):
        """Initialize with NumPy arrays"""
        self.use_torch = False
        
        self.x = np.zeros(self.n_states)
        
        self.F = np.eye(self.n_states)
        self.F[0:3, 3:6] = np.eye(3) * self.dt
        self.F[0:3, 6:9] = np.eye(3) * 0.5 * self.dt**2
        self.F[3:6, 6:9] = np.eye(3) * self.dt
        
        self.H = np.zeros((self.n_meas, self.n_states))
        self.H[0:3, 0:3] = np.eye(3)
        
        self.Q = np.eye(self.n_states) * pn
        self.Q[6:9, 6:9] *= 2
        self.R = np.eye(self.n_meas) * mn
        
        self.P = np.eye(self.n_states)
        self.initialized = False
    
    def initialize(self, position):
        if self.use_torch:
            self.x = torch.zeros(self.n_states, dtype=torch.float32, device=DEVICE)
            self.x[0:3] = torch.tensor(position, dtype=torch.float32, device=DEVICE)
        else:
            self.x = np.zeros(self.n_states)
            self.x[0:3] = position
        self.initialized = True
    
    def update(self, measurement):
        if not self.initialized:
            self.initialize(measurement)
            return
        
        if self.use_torch:
            self._update_torch(measurement)
        else:
            self._update_numpy(measurement)
    
    def _update_torch(self, measurement):
        z = torch.tensor(measurement, dtype=torch.float32, device=DEVICE)
        
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = torch.eye(self.n_states, dtype=torch.float32, device=DEVICE)
        self.P = (I - K @ self.H) @ self.P
    
    def _update_numpy(self, measurement):
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (measurement - self.H @ self.x)
        self.P = (np.eye(self.n_states) - K @ self.H) @ self.P
    
    def predict_future(self, steps: int) -> np.ndarray:
        if self.use_torch:
            predictions = []
            x_future = self.x.clone()
            for _ in range(steps):
                x_future = self.F @ x_future
                predictions.append(x_future[0:3].cpu().numpy())
            return np.array(predictions)
        else:
            predictions = []
            x_future = self.x.copy()
            for _ in range(steps):
                x_future = self.F @ x_future
                predictions.append(x_future[0:3].copy())
            return np.array(predictions)
    
    def get_state(self):
        if self.use_torch:
            return {
                'position': self.x[0:3].cpu().numpy(),
                'velocity': self.x[3:6].cpu().numpy(),
                'acceleration': self.x[6:9].cpu().numpy()
            }
        else:
            return {
                'position': self.x[0:3].copy(),
                'velocity': self.x[3:6].copy(),
                'acceleration': self.x[6:9].copy()
            }


# ============== Random Motion Obstacle ==============
class RandomMotionObstacle:
    """Randomly moving obstacle with boundary reflection"""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.dt = 0.1
        self.position = config.OBSTACLE_INITIAL_POS.copy()
        
        speed = np.random.uniform(*config.OBSTACLE_SPEED_RANGE)
        self.velocity = self._random_direction() * speed
        self.bounds = config.OBSTACLE_BOUNDS
    
    def _random_direction(self) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        return np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
    
    def step(self) -> np.ndarray:
        # Random speed adjustment
        speed = np.linalg.norm(self.velocity)
        speed += np.random.normal(0, self.cfg.OBSTACLE_SPEED_CHANGE_RATE * self.dt)
        speed = np.clip(speed, *self.cfg.OBSTACLE_SPEED_RANGE)
        
        # Random direction adjustment
        direction = self.velocity / (np.linalg.norm(self.velocity) + 1e-6)
        perturbation = np.random.normal(0, self.cfg.OBSTACLE_DIRECTION_CHANGE_RATE, 3)
        direction = direction + perturbation
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        self.velocity = direction * speed
        new_position = self.position + self.velocity * self.dt
        
        # Boundary reflection
        for i in range(3):
            if new_position[i] < self.bounds[i][0]:
                new_position[i] = self.bounds[i][0]
                self.velocity[i] = abs(self.velocity[i])
            elif new_position[i] > self.bounds[i][1]:
                new_position[i] = self.bounds[i][1]
                self.velocity[i] = -abs(self.velocity[i])
        
        self.position = new_position
        return self.position.copy()
    
    def reset(self):
        self.position = self.cfg.OBSTACLE_INITIAL_POS.copy()
        speed = np.random.uniform(*self.cfg.OBSTACLE_SPEED_RANGE)
        self.velocity = self._random_direction() * speed


# ============== GPU-Accelerated MPC Controller ==============
class FastMPCController:
    """High-performance MPC controller with optional GPU acceleration"""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.dt = 0.1
        
        self.horizon = config.HORIZON
        self.control_horizon = config.CONTROL_HORIZON
        self.max_iter = config.MPC_ITERATIONS
        
        self.max_vel = config.MAX_VELOCITY
        self.max_acc = config.MAX_ACCELERATION
        
        self.obstacle_radius = config.OBSTACLE_RADIUS
        self.safety_margin = config.SAFETY_MARGIN
        self.safe_dist = self.obstacle_radius + self.safety_margin
        
        # Cost weights
        self.w_goal = 1.0
        self.w_control = 0.1
        self.w_obs = 3000.0
        self.w_smooth = 0.05
        
        # State
        self.state = np.concatenate([config.START_POINT.copy(), np.zeros(3)])
        
        # Target path
        self.start_point = config.START_POINT.copy()
        self.end_point = config.END_POINT.copy()
        self.target_speed = config.TARGET_SPEED
        self.path_vector = self.end_point - self.start_point
        self.path_length = np.linalg.norm(self.path_vector)
        
        # Kalman filter
        self.kalman = KalmanFilter3D(
            config.KALMAN_PROCESS_NOISE,
            config.KALMAN_MEASUREMENT_NOISE
        )
        
        # Warm start
        self.prev_U = None
        
        # Precompute
        self.control_weights = np.exp(-0.1 * np.arange(self.control_horizon))
        
        # GPU tensors if available
        if TORCH_AVAILABLE:
            self._init_gpu_tensors()
    
    def _init_gpu_tensors(self):
        """Initialize GPU tensors for acceleration"""
        self.gpu_weights = torch.tensor(
            self.control_weights, dtype=torch.float32, device=DEVICE
        )
    
    def get_target_position(self, t: float) -> np.ndarray:
        total_distance = self.target_speed * t
        cycle_length = 2 * self.path_length
        distance_in_cycle = total_distance % cycle_length
        
        if distance_in_cycle <= self.path_length:
            ratio = distance_in_cycle / self.path_length
            return self.start_point + ratio * self.path_vector
        else:
            ratio = (distance_in_cycle - self.path_length) / self.path_length
            return self.end_point - ratio * self.path_vector
    
    def update_obstacle_observation(self, obs_position: np.ndarray):
        self.kalman.update(obs_position)
    
    def get_predicted_obstacle_trajectory(self) -> np.ndarray:
        return self.kalman.predict_future(self.horizon)
    
    def dynamics(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        pos, vel = state[:3], state[3:]
        acc = np.clip(u, -self.max_acc, self.max_acc)
        vel_new = np.clip(vel + acc * self.dt, -self.max_vel, self.max_vel)
        pos_new = pos + vel_new * self.dt
        return np.concatenate([pos_new, vel_new])
    
    def cost_function(self, U_flat: np.ndarray, state: np.ndarray,
                      goal_seq: np.ndarray, obs_seq: np.ndarray) -> float:
        """Cost function with optional GPU acceleration"""
        if TORCH_AVAILABLE and USE_GPU:
            return self._cost_function_gpu(U_flat, state, goal_seq, obs_seq)
        else:
            return self._cost_function_numpy(U_flat, state, goal_seq, obs_seq)
    
    def _cost_function_gpu(self, U_flat, state, goal_seq, obs_seq):
        """GPU-accelerated cost computation"""
        U = torch.tensor(U_flat.reshape(self.control_horizon, 3),
                        dtype=torch.float32, device=DEVICE)
        x = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        goals = torch.tensor(goal_seq, dtype=torch.float32, device=DEVICE)
        obs = torch.tensor(obs_seq, dtype=torch.float32, device=DEVICE)
        
        positions = torch.zeros(self.horizon, 3, dtype=torch.float32, device=DEVICE)
        
        for i in range(self.horizon):
            u = U[i] if i < self.control_horizon else torch.zeros(3, device=DEVICE)
            
            pos, vel = x[:3], x[3:]
            acc = torch.clamp(u, -self.max_acc, self.max_acc)
            vel_new = torch.clamp(vel + acc * self.dt, -self.max_vel, self.max_vel)
            pos_new = pos + vel_new * self.dt
            x = torch.cat([pos_new, vel_new])
            positions[i] = pos_new
        
        # Goal cost
        goal_cost = self.w_goal * torch.sum((positions - goals)**2)
        
        # Control cost
        control_cost = self.w_control * torch.sum(
            U**2 * self.gpu_weights.unsqueeze(1)
        )
        
        # Obstacle cost
        distances = torch.norm(positions - obs, dim=1)
        violations = torch.clamp(self.safe_dist - distances, min=0)
        obs_cost = self.w_obs * torch.sum(violations**2)
        
        # Smoothness cost
        if self.control_horizon > 1:
            u_diff = U[1:] - U[:-1]
            smooth_cost = self.w_smooth * torch.sum(u_diff**2)
        else:
            smooth_cost = 0
        
        total = goal_cost + control_cost + obs_cost + smooth_cost
        return total.cpu().item()
    
    def _cost_function_numpy(self, U_flat, state, goal_seq, obs_seq):
        """NumPy cost computation"""
        U = U_flat.reshape(self.control_horizon, 3)
        x = state.copy()
        positions = np.zeros((self.horizon, 3))
        
        for i in range(self.horizon):
            u = U[i] if i < self.control_horizon else np.zeros(3)
            x = self.dynamics(x, u)
            positions[i] = x[:3]
        
        # Costs
        goal_cost = self.w_goal * np.sum((positions - goal_seq)**2)
        control_cost = self.w_control * np.sum(U**2 * self.control_weights[:, np.newaxis])
        
        distances = np.linalg.norm(positions - obs_seq, axis=1)
        violations = np.maximum(0, self.safe_dist - distances)
        obs_cost = self.w_obs * np.sum(violations**2)
        
        if self.control_horizon > 1:
            smooth_cost = self.w_smooth * np.sum(np.diff(U, axis=0)**2)
        else:
            smooth_cost = 0
        
        return goal_cost + control_cost + obs_cost + smooth_cost
    
    def compute_control(self, t: float) -> np.ndarray:
        goal_seq = np.array([self.get_target_position(t + i * self.dt)
                           for i in range(self.horizon)])
        obs_seq = self.get_predicted_obstacle_trajectory()
        
        if self.prev_U is None:
            U0 = np.zeros(self.control_horizon * 3)
        else:
            U0 = np.roll(self.prev_U, -3)
            U0[-3:] = 0
        
        bounds = [(-self.max_acc, self.max_acc)] * (self.control_horizon * 3)
        
        result = minimize(
            self.cost_function, U0,
            args=(self.state, goal_seq, obs_seq),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.max_iter, 'disp': False, 'ftol': 1e-4}
        )
        
        self.prev_U = result.x
        return result.x[:3] if result.success else np.zeros(3)
    
    def step(self, t: float, obs_position: np.ndarray) -> np.ndarray:
        self.update_obstacle_observation(obs_position)
        u = self.compute_control(t)
        self.state = self.dynamics(self.state, u)
        return self.state[:3].copy()
    
    def reset(self):
        self.state = np.concatenate([self.cfg.START_POINT.copy(), np.zeros(3)])
        self.prev_U = None
        self.kalman = KalmanFilter3D(
            self.cfg.KALMAN_PROCESS_NOISE,
            self.cfg.KALMAN_MEASUREMENT_NOISE
        )
        if TORCH_AVAILABLE:
            self._init_gpu_tensors()


# ============== Async Simulator ==============
class AsyncSimulator:
    """Asynchronous simulator - separate computation and rendering"""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.dt = 0.1
        
        self.controller = FastMPCController(config)
        self.obstacle = RandomMotionObstacle(config)
        
        self.state_lock = threading.Lock()
        self.current_state = {
            'robot_pos': config.START_POINT.copy(),
            'goal_pos': config.START_POINT.copy(),
            'obs_pos': config.OBSTACLE_INITIAL_POS.copy(),
            'predicted_obs': np.zeros((config.HORIZON, 3)),
            'kalman_state': {'velocity': np.zeros(3), 'acceleration': np.zeros(3)},
            'time': 0.0,
            'dist_to_obs': 10.0,
            'dist_to_goal': 0.0
        }
        
        self.history_lock = threading.Lock()
        self.robot_history = []
        self.goal_history = []
        self.obs_history = []
        
        self.running = False
        self.paused = False
        self.t = 0.0
        
        self.compute_thread: Optional[threading.Thread] = None
    
    def _compute_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            
            start_time = time.perf_counter()
            
            obs_pos = self.obstacle.step()
            robot_pos = self.controller.step(self.t, obs_pos)
            goal_pos = self.controller.get_target_position(self.t)
            
            predicted_obs = self.controller.get_predicted_obstacle_trajectory()
            kalman_state = self.controller.kalman.get_state()
            
            dist_to_obs = np.linalg.norm(robot_pos - obs_pos)
            dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
            
            with self.state_lock:
                self.current_state = {
                    'robot_pos': robot_pos,
                    'goal_pos': goal_pos,
                    'obs_pos': obs_pos,
                    'predicted_obs': predicted_obs,
                    'kalman_state': kalman_state,
                    'time': self.t,
                    'dist_to_obs': dist_to_obs,
                    'dist_to_goal': dist_to_goal
                }
            
            with self.history_lock:
                self.robot_history.append(robot_pos.copy())
                self.goal_history.append(goal_pos.copy())
                self.obs_history.append(obs_pos.copy())
                
                while len(self.robot_history) > self.cfg.ROBOT_TRAIL_LENGTH:
                    self.robot_history.pop(0)
                while len(self.goal_history) > self.cfg.GOAL_TRAIL_LENGTH:
                    self.goal_history.pop(0)
                while len(self.obs_history) > self.cfg.OBS_TRAIL_LENGTH:
                    self.obs_history.pop(0)
            
            self.t += self.dt
            
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, self.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start(self):
        self.running = True
        self.compute_thread = threading.Thread(target=self._compute_loop, daemon=True)
        self.compute_thread.start()
    
    def stop(self):
        self.running = False
        if self.compute_thread:
            self.compute_thread.join(timeout=1.0)
    
    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused
    
    def reset(self):
        was_paused = self.paused
        self.paused = True
        time.sleep(0.05)
        
        self.t = 0.0
        self.controller.reset()
        self.obstacle.reset()
        
        with self.history_lock:
            self.robot_history.clear()
            self.goal_history.clear()
            self.obs_history.clear()
        
        with self.state_lock:
            self.current_state = {
                'robot_pos': self.cfg.START_POINT.copy(),
                'goal_pos': self.cfg.START_POINT.copy(),
                'obs_pos': self.cfg.OBSTACLE_INITIAL_POS.copy(),
                'predicted_obs': np.zeros((self.cfg.HORIZON, 3)),
                'kalman_state': {'velocity': np.zeros(3), 'acceleration': np.zeros(3)},
                'time': 0.0,
                'dist_to_obs': 10.0,
                'dist_to_goal': 0.0
            }
        
        self.paused = was_paused
    
    def get_state(self):
        with self.state_lock:
            return self.current_state.copy()
    
    def get_histories(self):
        with self.history_lock:
            return (
                list(self.robot_history),
                list(self.goal_history),
                list(self.obs_history)
            )


# ============== VTK Visualizer ==============
class VTKVisualizer:
    """VTK visualizer with English-only text (no font issues)"""
    
    def __init__(self, simulator: AsyncSimulator):
        self.sim = simulator
        self.cfg = simulator.cfg
        
        self.renderer = vtk.vtkRenderer()
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(*self.cfg.WINDOW_SIZE)
        self.window.SetWindowName(
            "3D MPC Obstacle Avoidance (Kalman Prediction) - Space:Pause R:Reset Q:Quit"
        )
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        self._setup_scene()
        
        self.interactor.AddObserver('TimerEvent', self._on_timer)
        self.interactor.AddObserver('KeyPressEvent', self._on_key)
    
    def _create_sphere(self, radius, color, opacity=1.0):
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
    
    def _create_prediction_trail(self, color):
        actors = []
        for _ in range(self.cfg.PREDICTION_TRAIL_LENGTH):
            actor = self._create_sphere(0.08, color, 0.6)
            actor.VisibilityOff()
            self.renderer.AddActor(actor)
            actors.append(actor)
        return actors
    
    def _create_axes_without_labels(self):
        """Create coordinate axes without text labels"""
        # X axis (red)
        x_line = vtk.vtkLineSource()
        x_line.SetPoint1(0, 0, 0)
        x_line.SetPoint2(2, 0, 0)
        x_mapper = vtk.vtkPolyDataMapper()
        x_mapper.SetInputConnection(x_line.GetOutputPort())
        x_actor = vtk.vtkActor()
        x_actor.SetMapper(x_mapper)
        x_actor.GetProperty().SetColor(1, 0, 0)
        x_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(x_actor)
        
        # Y axis (green)
        y_line = vtk.vtkLineSource()
        y_line.SetPoint1(0, 0, 0)
        y_line.SetPoint2(0, 2, 0)
        y_mapper = vtk.vtkPolyDataMapper()
        y_mapper.SetInputConnection(y_line.GetOutputPort())
        y_actor = vtk.vtkActor()
        y_actor.SetMapper(y_mapper)
        y_actor.GetProperty().SetColor(0, 1, 0)
        y_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(y_actor)
        
        # Z axis (blue)
        z_line = vtk.vtkLineSource()
        z_line.SetPoint1(0, 0, 0)
        z_line.SetPoint2(0, 0, 2)
        z_mapper = vtk.vtkPolyDataMapper()
        z_mapper.SetInputConnection(z_line.GetOutputPort())
        z_actor = vtk.vtkActor()
        z_actor.SetMapper(z_mapper)
        z_actor.GetProperty().SetColor(0, 0, 1)
        z_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(z_actor)
    
    def _setup_scene(self):
        # Background
        self.renderer.SetBackground(0.05, 0.05, 0.1)
        self.renderer.SetBackground2(0.15, 0.15, 0.25)
        self.renderer.GradientBackgroundOn()
        
        # Start/End markers (simple spheres without text)
        start_marker = self._create_sphere(0.2, (0.2, 0.9, 0.2), 0.8)
        start_marker.SetPosition(self.cfg.START_POINT)
        self.renderer.AddActor(start_marker)
        
        end_marker = self._create_sphere(0.2, (0.9, 0.2, 0.9), 0.8)
        end_marker.SetPosition(self.cfg.END_POINT)
        self.renderer.AddActor(end_marker)
        
        # Path line
        self._create_path_line()
        
        # Robot
        self.robot_actor = self._create_sphere(self.cfg.ROBOT_RADIUS, (0.2, 0.6, 1.0))
        self.renderer.AddActor(self.robot_actor)
        
        # Obstacle
        self.obs_actor = self._create_sphere(self.cfg.OBSTACLE_RADIUS, (1.0, 0.3, 0.3), 0.85)
        self.renderer.AddActor(self.obs_actor)
        
        # Safety boundary
        safe_r = self.cfg.OBSTACLE_RADIUS + self.cfg.SAFETY_MARGIN
        self.safety_actor = self._create_sphere(safe_r, (1.0, 0.5, 0.5), 0.15)
        self.renderer.AddActor(self.safety_actor)
        
        # Goal point
        self.goal_actor = self._create_sphere(0.2, (0.3, 1.0, 0.4))
        self.renderer.AddActor(self.goal_actor)
        
        # Trails
        (self.robot_trail_actor, self.robot_trail_points,
         self.robot_trail_lines, self.robot_trail_polydata,
         self.robot_trail_tube) = self._create_trail((0.3, 0.7, 1.0), 0.05)
        self.renderer.AddActor(self.robot_trail_actor)
        
        (self.goal_trail_actor, self.goal_trail_points,
         self.goal_trail_lines, self.goal_trail_polydata,
         self.goal_trail_tube) = self._create_trail((0.4, 1.0, 0.4), 0.025, 0.6)
        self.renderer.AddActor(self.goal_trail_actor)
        
        (self.obs_trail_actor, self.obs_trail_points,
         self.obs_trail_lines, self.obs_trail_polydata,
         self.obs_trail_tube) = self._create_trail((1.0, 0.5, 0.5), 0.02, 0.5)
        self.renderer.AddActor(self.obs_trail_actor)
        
        # Kalman prediction trail (yellow dots)
        self.prediction_actors = self._create_prediction_trail((1.0, 0.9, 0.2))
        
        # Ground grid
        self._create_ground()
        
        # Axes without labels
        self._create_axes_without_labels()
        
        # Text displays (English only)
        self._create_texts()
        
        # Lighting
        self._setup_lights()
        
        # Camera
        self._setup_camera()
    
    def _create_path_line(self):
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
    
    def _create_texts(self):
        """Create text displays - English only to avoid font issues"""
        # Status info
        self.info_text = vtk.vtkTextActor()
        self.info_text.GetTextProperty().SetFontSize(14)
        self.info_text.GetTextProperty().SetColor(1, 1, 1)
        self.info_text.GetTextProperty().SetFontFamilyToCourier()
        self.info_text.SetPosition(20, 20)
        self.renderer.AddActor2D(self.info_text)
        
        # Title
        title = vtk.vtkTextActor()
        gpu_str = "GPU" if USE_GPU else "CPU"
        title.SetInput(f"3D MPC Obstacle Avoidance - Kalman Prediction [{gpu_str}]")
        title.GetTextProperty().SetFontSize(22)
        title.GetTextProperty().SetColor(0.9, 0.9, 1.0)
        title.GetTextProperty().SetBold(True)
        title.GetTextProperty().SetFontFamilyToCourier()
        title.SetPosition(20, 910)
        self.renderer.AddActor2D(title)
        
        # Legend
        legend = vtk.vtkTextActor()
        legend.SetInput(
            "Legend:\n"
            "* Blue  - Robot\n"
            "* Red   - Obstacle (random)\n"
            "* Green - Target\n"
            "* Yellow- Kalman prediction\n"
            "* Trans - Safety boundary\n\n"
            "Controls:\n"
            "Space - Pause/Resume\n"
            "R     - Reset\n"
            "Q     - Quit"
        )
        legend.GetTextProperty().SetFontSize(12)
        legend.GetTextProperty().SetColor(0.85, 0.85, 0.9)
        legend.GetTextProperty().SetFontFamilyToCourier()
        legend.SetPosition(1280, 20)
        self.renderer.AddActor2D(legend)
    
    def _setup_lights(self):
        light1 = vtk.vtkLight()
        light1.SetPosition(15, 15, 20)
        light1.SetIntensity(0.8)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetPosition(-10, -10, 15)
        light2.SetIntensity(0.4)
        self.renderer.AddLight(light2)
    
    def _setup_camera(self):
        center = (self.cfg.START_POINT + self.cfg.END_POINT) / 2
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0] + 18, center[1] + 18, center[2] + 14)
        camera.SetFocalPoint(center)
        camera.SetViewUp(0, 0, 1)
    
    def _update_trail(self, points, lines, polydata, history):
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
    
    def _on_timer(self, obj, event):
        state = self.sim.get_state()
        robot_history, goal_history, obs_history = self.sim.get_histories()
        
        # Update positions
        self.robot_actor.SetPosition(state['robot_pos'])
        self.goal_actor.SetPosition(state['goal_pos'])
        self.obs_actor.SetPosition(state['obs_pos'])
        self.safety_actor.SetPosition(state['obs_pos'])
        
        # Update trails
        self._update_trail(self.robot_trail_points, self.robot_trail_lines,
                          self.robot_trail_polydata, robot_history)
        self.robot_trail_tube.Modified()
        
        self._update_trail(self.goal_trail_points, self.goal_trail_lines,
                          self.goal_trail_polydata, goal_history)
        self.goal_trail_tube.Modified()
        
        self._update_trail(self.obs_trail_points, self.obs_trail_lines,
                          self.obs_trail_polydata, obs_history)
        self.obs_trail_tube.Modified()
        
        # Update prediction trail
        predicted = state['predicted_obs']
        for i, actor in enumerate(self.prediction_actors):
            if i < len(predicted):
                actor.SetPosition(predicted[i])
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        
        # Robot color based on distance
        safe_dist = self.cfg.OBSTACLE_RADIUS + self.cfg.SAFETY_MARGIN
        if state['dist_to_obs'] < safe_dist * 1.3:
            danger = min(1, (safe_dist * 1.3 - state['dist_to_obs']) / (safe_dist * 0.3))
            self.robot_actor.GetProperty().SetColor(
                0.2 + 0.8 * danger, 0.6 * (1 - danger), 1.0 * (1 - danger)
            )
        else:
            self.robot_actor.GetProperty().SetColor(0.2, 0.6, 1.0)
        
        # Update info text (English only)
        ks = state['kalman_state']
        vel = ks['velocity']
        acc = ks['acceleration']
        speed = np.linalg.norm(vel)
        
        status = "!! AVOIDING !!" if state['dist_to_obs'] < safe_dist * 1.2 else "OK Safe"
        paused_str = " [PAUSED]" if self.sim.paused else ""
        
        self.info_text.SetInput(
            f"Time: {state['time']:.1f}s{paused_str}\n"
            f"Robot:    ({state['robot_pos'][0]:6.2f}, {state['robot_pos'][1]:6.2f}, {state['robot_pos'][2]:6.2f})\n"
            f"Obstacle: ({state['obs_pos'][0]:6.2f}, {state['obs_pos'][1]:6.2f}, {state['obs_pos'][2]:6.2f})\n"
            f"Dist to Obs: {state['dist_to_obs']:.2f}m | Dist to Goal: {state['dist_to_goal']:.2f}m\n"
            f"Safe dist: {safe_dist:.2f}m | Status: {status}\n"
            f"------- Kalman Estimation -------\n"
            f"Obs Velocity: ({vel[0]:5.2f}, {vel[1]:5.2f}, {vel[2]:5.2f}) |v|={speed:.2f} m/s\n"
            f"Obs Accel:    ({acc[0]:5.2f}, {acc[1]:5.2f}, {acc[2]:5.2f})"
        )
        
        self.window.Render()
    
    def _on_key(self, obj, event):
        key = self.interactor.GetKeySym().lower()
        
        if key == 'space':
            paused = self.sim.toggle_pause()
            print(f"[{'Paused' if paused else 'Running'}]")
        elif key == 'r':
            self.sim.reset()
            print("[Reset]")
        elif key == 'q':
            print("[Quit]")
            self.sim.stop()
            self.interactor.TerminateApp()
    
    def run(self):
        print("=" * 65)
        print("3D MPC Obstacle Avoidance - Kalman Prediction")
        print("=" * 65)
        
        if USE_GPU:
            print(f"\n[GPU Mode] Device: {DEVICE}")
        else:
            print(f"\n[CPU Mode] {'PyTorch optimized' if TORCH_AVAILABLE else 'NumPy only'}")
        
        print(f"\nConfiguration:")
        print(f"  Target speed: {self.cfg.TARGET_SPEED} m/s")
        print(f"  Obstacle speed range: {self.cfg.OBSTACLE_SPEED_RANGE} m/s")
        print(f"  Safety distance: {self.cfg.OBSTACLE_RADIUS + self.cfg.SAFETY_MARGIN} m")
        print(f"\nFeatures:")
        print("  * Multi-threaded: View rotation doesn't affect simulation")
        print("  * Kalman filter: Real-time obstacle trajectory prediction")
        print("  * Random motion: Obstacle with variable speed/direction")
        print(f"\nControls: Space=Pause  R=Reset  Q=Quit")
        print("\nStarting...\n")
        
        self.sim.start()
        
        self.window.Render()
        self.interactor.CreateRepeatingTimer(33)  # ~30 FPS
        self.interactor.Start()
        
        self.sim.stop()


def main():
    # Configuration
    cfg.START_POINT = np.array([0.0, 0.0, 0.0])
    cfg.END_POINT = np.array([10.0, 8.0, 6.0])
    cfg.TARGET_SPEED = 1.2
    
    # Obstacle random motion
    cfg.OBSTACLE_INITIAL_POS = np.array([5.0, 4.0, 3.0])
    cfg.OBSTACLE_SPEED_RANGE = (0.8, 2.5)
    cfg.OBSTACLE_SPEED_CHANGE_RATE = 0.5
    cfg.OBSTACLE_DIRECTION_CHANGE_RATE = 0.15
    cfg.OBSTACLE_BOUNDS = ((-1, 4), (-1, 4), (0, 4))
    
    # Trail lengths
    cfg.ROBOT_TRAIL_LENGTH = 60
    cfg.GOAL_TRAIL_LENGTH = 60
    cfg.OBS_TRAIL_LENGTH = 40
    cfg.PREDICTION_TRAIL_LENGTH = 15
    
    # Kalman filter
    cfg.KALMAN_PROCESS_NOISE = 0.8
    cfg.KALMAN_MEASUREMENT_NOISE = 0.1
    
    # Run
    simulator = AsyncSimulator(cfg)
    visualizer = VTKVisualizer(simulator)
    visualizer.run()


if __name__ == "__main__":
    main()
