
"""
3D MPC 轨迹跟踪与动态避障 - VTK实时可视化 (改进版)
====================================================
功能：
- 目标点在起点和终点之间往返运动，可设置速度
- 障碍物（红色球体）在3D空间中做李萨如曲线运动
- MPC控制器实时计算避障路径
- 清晰显示目标轨迹（绿色）和实际运动轨迹（蓝色）

运行方式：python vtk_3d_obstacle_avoidance_v2.py
"""

import numpy as np
from scipy.optimize import minimize
import vtk


# ============== 可配置参数 ==============
class Config:
    """
    配置参数 - 可根据需要修改
    
    轨迹长度说明：
    - ROBOT_TRAIL_LENGTH: 机器人实际运动轨迹保留的点数
    - GOAL_TRAIL_LENGTH:  目标点轨迹保留的点数
    - OBS_TRAIL_LENGTH:   障碍物轨迹保留的点数
    
    轨迹长度 = 点数 × 时间步长(0.1s)
    例如: 500点 = 50秒的轨迹
    """
    
    # 目标运动参数
    START_POINT = np.array([0.0, 0.0, 0.0])      # 起始位置
    END_POINT = np.array([8.0, 6.0, 5.0])        # 终点位置
    TARGET_SPEED = 1.0                            # 目标运动速度 (m/s)
    
    # 障碍物参数
    OBSTACLE_CENTER = np.array([4.0, 3.0, 2.5])  # 障碍物运动中心
    OBSTACLE_AMPLITUDE = np.array([2.0, 2.0, 1.5])  # 运动幅度 (X, Y, Z方向)
    OBSTACLE_FREQ = np.array([0.25, 0.3, 0.35])  # 各轴运动频率 (rad/s)
    OBSTACLE_RADIUS = 0.8                         # 障碍物半径
    SAFETY_MARGIN = 0.4                           # 安全边距
    
    # 机器人参数
    ROBOT_RADIUS = 0.35                           # 机器人显示半径
    MAX_VELOCITY = 2.5                            # 最大速度
    MAX_ACCELERATION = 2.0                        # 最大加速度
    
    # MPC参数
    HORIZON = 12                                  # 预测时域
    CONTROL_HORIZON = 6                           # 控制时域
    
    # ====== 轨迹长度设置 ======
    ROBOT_TRAIL_LENGTH = 50                      # 机器人轨迹长度 (点数, 500点=50秒)
    GOAL_TRAIL_LENGTH = 50                      # 目标轨迹长度 (点数)
    OBS_TRAIL_LENGTH = 30                        # 障碍物轨迹长度 (点数)
    
    # 显示参数
    WINDOW_SIZE = (1400, 900)                     # 窗口大小


class MPC3DController:
    """3D MPC控制器"""
    
    def __init__(self, config=Config):
        self.cfg = config
        self.dt = 0.1
        
        # MPC参数
        self.horizon = config.HORIZON
        self.control_horizon = config.CONTROL_HORIZON
        self.max_iter = 50
        
        # 约束
        self.max_vel = config.MAX_VELOCITY
        self.max_acc = config.MAX_ACCELERATION
        
        # 障碍物参数
        self.obstacle_radius = config.OBSTACLE_RADIUS
        self.safety_margin = config.SAFETY_MARGIN
        
        # 权重
        self.w_goal = 1.0
        self.w_control = 0.1
        self.w_obs = 2000.0
        
        # 状态 [x, y, z, vx, vy, vz]
        self.state = np.concatenate([config.START_POINT, [0.0, 0.0, 0.0]])
        
        # 目标往返运动参数
        self.start_point = config.START_POINT.copy()
        self.end_point = config.END_POINT.copy()
        self.target_speed = config.TARGET_SPEED
        self.direction = 1  # 1: 去程, -1: 返程
        self.target_position = self.start_point.copy()
        
        # 计算往返路径
        self.path_vector = self.end_point - self.start_point
        self.path_length = np.linalg.norm(self.path_vector)
        self.path_direction = self.path_vector / self.path_length
        self.current_distance = 0.0  # 沿路径的当前距离
        
        # warm start
        self.prev_U = None
    
    def get_target_position(self, t):
        """获取目标位置 - 往返运动"""
        # 计算沿路径移动的距离
        total_distance = self.target_speed * t
        
        # 计算往返周期内的位置
        cycle_length = 2 * self.path_length  # 一个完整往返的距离
        distance_in_cycle = total_distance % cycle_length
        
        if distance_in_cycle <= self.path_length:
            # 去程
            ratio = distance_in_cycle / self.path_length
            position = self.start_point + ratio * self.path_vector
        else:
            # 返程
            ratio = (distance_in_cycle - self.path_length) / self.path_length
            position = self.end_point - ratio * self.path_vector
        
        return position
    
    def obstacle_position(self, t):
        """障碍物运动轨迹 - 3D李萨如曲线"""
        cfg = self.cfg
        x = cfg.OBSTACLE_CENTER[0] + cfg.OBSTACLE_AMPLITUDE[0] * np.cos(cfg.OBSTACLE_FREQ[0] * t)
        y = cfg.OBSTACLE_CENTER[1] + cfg.OBSTACLE_AMPLITUDE[1] * np.sin(cfg.OBSTACLE_FREQ[1] * t)
        z = cfg.OBSTACLE_CENTER[2] + cfg.OBSTACLE_AMPLITUDE[2] * np.sin(cfg.OBSTACLE_FREQ[2] * t)
        return np.array([x, y, z])
    
    def dynamics(self, state, u):
        """机器人动力学模型"""
        pos = state[:3]
        vel = state[3:]
        acc = np.clip(u, -self.max_acc, self.max_acc)
        
        vel_new = np.clip(vel + acc * self.dt, -self.max_vel, self.max_vel)
        pos_new = pos + vel_new * self.dt
        
        return np.concatenate([pos_new, vel_new])
    
    def cost_function(self, U_flat, state, goal_seq, obs_seq):
        """MPC代价函数"""
        U = U_flat.reshape(-1, 3)
        x = state.copy()
        cost = 0.0
        
        for i in range(self.horizon):
            u = U[i] if i < self.control_horizon else np.zeros(3)
            x = self.dynamics(x, u)
            
            # 目标跟踪
            cost += self.w_goal * np.sum((x[:3] - goal_seq[i])**2)
            
            # 控制代价
            if i < self.control_horizon:
                cost += self.w_control * np.sum(u**2)
            
            # 障碍物避让
            dist = np.linalg.norm(x[:3] - obs_seq[i])
            safe_dist = self.obstacle_radius + self.safety_margin
            if dist < safe_dist:
                cost += self.w_obs * (safe_dist - dist)**2
        
        return cost
    
    def compute_control(self, t):
        """计算最优控制输入"""
        # 生成预测序列
        goal_seq = np.array([self.get_target_position(t + i*self.dt) 
                           for i in range(self.horizon)])
        obs_seq = np.array([self.obstacle_position(t + i*self.dt) 
                          for i in range(self.horizon)])
        
        # 初始猜测
        if self.prev_U is None:
            U0 = np.zeros(self.control_horizon * 3)
        else:
            U0 = np.roll(self.prev_U, -3)
            U0[-3:] = 0
        
        # 优化
        bounds = [(-self.max_acc, self.max_acc)] * (self.control_horizon * 3)
        result = minimize(
            self.cost_function, U0,
            args=(self.state, goal_seq, obs_seq),
            method='SLSQP', bounds=bounds,
            options={'maxiter': self.max_iter, 'disp': False}
        )
        
        self.prev_U = result.x
        return result.x[:3] if result.success else np.zeros(3)
    
    def step(self, t):
        """执行一步仿真"""
        u = self.compute_control(t)
        self.state = self.dynamics(self.state, u)
        return self.state[:3].copy()
    
    def reset(self):
        """重置状态"""
        self.state = np.concatenate([self.cfg.START_POINT, [0.0, 0.0, 0.0]])
        self.prev_U = None


class VTKVisualizer:
    """VTK实时可视化器"""
    
    def __init__(self, config=Config):
        self.cfg = config
        
        # MPC控制器
        self.controller = MPC3DController(config)
        
        # 时间
        self.t = 0.0
        self.dt = 0.1
        self.paused = False
        
        # 轨迹历史 - 分别设置长度
        self.robot_history = []
        self.goal_history = []
        self.obs_history = []
        self.robot_trail_length = config.ROBOT_TRAIL_LENGTH
        self.goal_trail_length = config.GOAL_TRAIL_LENGTH
        self.obs_trail_length = config.OBS_TRAIL_LENGTH
        
        # 创建渲染器和窗口
        self.renderer = vtk.vtkRenderer()
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(config.WINDOW_SIZE)
        self.window.SetWindowName("3D MPC 动态避障 - 空格:暂停 R:重置 Q:退出")
        
        # 交互器
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # 初始化场景
        self._setup_scene()
        
        # 设置回调
        self.interactor.AddObserver('TimerEvent', self._on_timer)
        self.interactor.AddObserver('KeyPressEvent', self._on_key)
    
    def _create_sphere_actor(self, radius, color, opacity=1.0):
        """创建球体Actor"""
        source = vtk.vtkSphereSource()
        source.SetRadius(radius)
        source.SetThetaResolution(32)
        source.SetPhiResolution(32)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(20)
        
        return actor
    
    def _create_trail_actor(self, color, radius=0.05, opacity=1.0):
        """创建轨迹管状Actor"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(12)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        
        return actor, points, lines, polydata, tube
    
    def _create_path_line(self, start, end, color, line_width=2, stipple=True):
        """创建路径指示线"""
        points = vtk.vtkPoints()
        points.InsertNextPoint(start)
        points.InsertNextPoint(end)
        
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
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetOpacity(0.6)
        
        return actor
    
    def _create_marker(self, position, color, label):
        """创建位置标记（球体+文字）"""
        # 小球
        sphere = self._create_sphere_actor(0.2, color, 0.8)
        sphere.SetPosition(position)
        self.renderer.AddActor(sphere)
        
        # 文字标签
        text = vtk.vtkBillboardTextActor3D()
        text.SetInput(label)
        text.SetPosition(position[0], position[1], position[2] + 0.5)
        text.GetTextProperty().SetFontSize(20)
        text.GetTextProperty().SetColor(color)
        text.GetTextProperty().SetBold(True)
        self.renderer.AddActor(text)
        
        return sphere
    
    def _setup_scene(self):
        """设置场景"""
        cfg = self.cfg
        
        # 背景渐变
        self.renderer.SetBackground(0.08, 0.08, 0.12)
        self.renderer.SetBackground2(0.2, 0.2, 0.3)
        self.renderer.GradientBackgroundOn()
        
        # === 起点和终点标记 ===
        self._create_marker(cfg.START_POINT, (0.2, 0.8, 0.2), "起点")
        self._create_marker(cfg.END_POINT, (0.8, 0.2, 0.8), "终点")
        
        # === 目标路径线（虚线效果用多段实现）===
        self.path_line = self._create_path_line(
            cfg.START_POINT, cfg.END_POINT, (0.5, 1.0, 0.5), 3)
        self.renderer.AddActor(self.path_line)
        
        # === 机器人 (蓝色球体) ===
        self.robot_actor = self._create_sphere_actor(cfg.ROBOT_RADIUS, (0.2, 0.6, 1.0))
        self.robot_actor.SetPosition(cfg.START_POINT)
        self.renderer.AddActor(self.robot_actor)
        
        # === 障碍物 (红色球体) ===
        self.obstacle_actor = self._create_sphere_actor(
            cfg.OBSTACLE_RADIUS, (1.0, 0.3, 0.3), 0.85)
        self.renderer.AddActor(self.obstacle_actor)
        
        # === 安全边界 (透明红色) ===
        safe_radius = cfg.OBSTACLE_RADIUS + cfg.SAFETY_MARGIN
        self.safety_actor = self._create_sphere_actor(safe_radius, (1.0, 0.5, 0.5), 0.15)
        self.renderer.AddActor(self.safety_actor)
        
        # === 目标点 (绿色球体) ===
        self.goal_actor = self._create_sphere_actor(0.25, (0.3, 1.0, 0.4))
        self.renderer.AddActor(self.goal_actor)
        
        # === 轨迹线 ===
        # 实际运动轨迹 (蓝色，粗)
        (self.robot_trail_actor, self.robot_trail_points, 
         self.robot_trail_lines, self.robot_trail_polydata,
         self.robot_trail_tube) = self._create_trail_actor((0.3, 0.7, 1.0), 0.06)
        self.renderer.AddActor(self.robot_trail_actor)
        
        # 目标轨迹 (绿色，细，半透明)
        (self.goal_trail_actor, self.goal_trail_points,
         self.goal_trail_lines, self.goal_trail_polydata,
         self.goal_trail_tube) = self._create_trail_actor((0.4, 1.0, 0.4), 0.03, 0.7)
        self.renderer.AddActor(self.goal_trail_actor)
        
        # 障碍物轨迹 (红色，更细，半透明)
        (self.obs_trail_actor, self.obs_trail_points,
         self.obs_trail_lines, self.obs_trail_polydata,
         self.obs_trail_tube) = self._create_trail_actor((1.0, 0.5, 0.5), 0.02, 0.5)
        self.renderer.AddActor(self.obs_trail_actor)
        
        # === 地面网格 ===
        self._create_ground_grid()
        
        # === 坐标轴 ===
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2, 2, 2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
        self.renderer.AddActor(axes)
        
        # === 信息文本 ===
        self._create_text_displays()
        
        # === 光源 ===
        light = vtk.vtkLight()
        light.SetPosition(15, 15, 20)
        light.SetFocalPoint(4, 3, 2)
        light.SetIntensity(0.9)
        self.renderer.AddLight(light)
        
        fill_light = vtk.vtkLight()
        fill_light.SetPosition(-10, -10, 15)
        fill_light.SetIntensity(0.4)
        self.renderer.AddLight(fill_light)
        
        # === 相机 ===
        self._setup_camera()
    
    def _create_ground_grid(self):
        """创建地面参考网格"""
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-2, -2, -0.1)
        plane.SetPoint1(12, -2, -0.1)
        plane.SetPoint2(-2, 10, -0.1)
        plane.SetXResolution(14)
        plane.SetYResolution(12)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.4, 0.4, 0.45)
        actor.GetProperty().SetOpacity(0.3)
        
        self.renderer.AddActor(actor)
    
    def _create_text_displays(self):
        """创建文本显示"""
        cfg = self.cfg
        
        # 状态信息
        self.info_text = vtk.vtkTextActor()
        self.info_text.SetInput("初始化中...")
        self.info_text.GetTextProperty().SetFontSize(16)
        self.info_text.GetTextProperty().SetColor(1, 1, 1)
        self.info_text.SetPosition(20, 20)
        self.renderer.AddActor2D(self.info_text)
        
        # 标题
        title = vtk.vtkTextActor()
        title.SetInput("3D MPC 轨迹跟踪与动态避障")
        title.GetTextProperty().SetFontSize(24)
        title.GetTextProperty().SetColor(0.9, 0.9, 1.0)
        title.GetTextProperty().SetBold(True)
        title.SetPosition(20, 860)
        self.renderer.AddActor2D(title)
        
        # 配置信息
        config_text = vtk.vtkTextActor()
        config_text.SetInput(
            f"配置参数:\n"
            f"起点: ({cfg.START_POINT[0]:.1f}, {cfg.START_POINT[1]:.1f}, {cfg.START_POINT[2]:.1f})\n"
            f"终点: ({cfg.END_POINT[0]:.1f}, {cfg.END_POINT[1]:.1f}, {cfg.END_POINT[2]:.1f})\n"
            f"目标速度: {cfg.TARGET_SPEED:.1f} m/s\n"
            f"障碍物半径: {cfg.OBSTACLE_RADIUS:.1f} m\n"
            f"安全边距: {cfg.SAFETY_MARGIN:.1f} m\n"
            f"轨迹长度: 机器人{cfg.ROBOT_TRAIL_LENGTH} 目标{cfg.GOAL_TRAIL_LENGTH} 障碍{cfg.OBS_TRAIL_LENGTH}"
        )
        config_text.GetTextProperty().SetFontSize(13)
        config_text.GetTextProperty().SetColor(0.8, 0.8, 0.9)
        config_text.SetPosition(20, 720)
        self.renderer.AddActor2D(config_text)
        
        # 图例
        legend = vtk.vtkTextActor()
        legend.SetInput(
            "图例:\n"
            "● 蓝色球/线 - 机器人及轨迹\n"
            "● 绿色球/线 - 目标及轨迹\n"
            "● 红色球   - 障碍物\n"
            "● 透明球   - 安全边界\n"
            "━ 绿色直线 - 目标路径\n\n"
            "操作: 空格-暂停 R-重置 Q-退出"
        )
        legend.GetTextProperty().SetFontSize(13)
        legend.GetTextProperty().SetColor(0.85, 0.85, 0.9)
        legend.SetPosition(1200, 20)
        self.renderer.AddActor2D(legend)
    
    def _setup_camera(self):
        """设置相机视角"""
        cfg = self.cfg
        # 计算场景中心
        center = (cfg.START_POINT + cfg.END_POINT) / 2
        
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0] + 15, center[1] + 15, center[2] + 12)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
    
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
    
    def _on_timer(self, obj, event):
        """定时器回调 - 更新动画"""
        if self.paused:
            return
        
        # 获取当前位置
        robot_pos = self.controller.step(self.t)
        goal_pos = self.controller.get_target_position(self.t)
        obs_pos = self.controller.obstacle_position(self.t)
        
        # 更新历史
        self.robot_history.append(robot_pos.copy())
        self.goal_history.append(goal_pos.copy())
        self.obs_history.append(obs_pos.copy())
        
        # 限制各轨迹长度（分别控制）
        while len(self.robot_history) > self.robot_trail_length:
            self.robot_history.pop(0)
        while len(self.goal_history) > self.goal_trail_length:
            self.goal_history.pop(0)
        while len(self.obs_history) > self.obs_trail_length:
            self.obs_history.pop(0)
        
        # 更新Actor位置
        self.robot_actor.SetPosition(robot_pos)
        self.goal_actor.SetPosition(goal_pos)
        self.obstacle_actor.SetPosition(obs_pos)
        self.safety_actor.SetPosition(obs_pos)
        
        # 更新轨迹
        self._update_trail(self.robot_trail_points, self.robot_trail_lines,
                          self.robot_trail_polydata, self.robot_history)
        self.robot_trail_tube.Modified()
        
        self._update_trail(self.goal_trail_points, self.goal_trail_lines,
                          self.goal_trail_polydata, self.goal_history)
        self.goal_trail_tube.Modified()
        
        self._update_trail(self.obs_trail_points, self.obs_trail_lines,
                          self.obs_trail_polydata, self.obs_history)
        self.obs_trail_tube.Modified()
        
        # 计算距离和状态
        dist_to_obs = np.linalg.norm(robot_pos - obs_pos)
        dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
        safe_dist = self.cfg.OBSTACLE_RADIUS + self.cfg.SAFETY_MARGIN
        
        # 判断运动方向
        path_length = np.linalg.norm(self.cfg.END_POINT - self.cfg.START_POINT)
        total_dist = self.cfg.TARGET_SPEED * self.t
        cycle_dist = total_dist % (2 * path_length)
        direction = "→ 去程" if cycle_dist <= path_length else "← 返程"
        
        # 根据距离改变机器人颜色
        if dist_to_obs < safe_dist * 1.3:
            danger = min(1.0, (safe_dist * 1.3 - dist_to_obs) / (safe_dist * 0.3))
            self.robot_actor.GetProperty().SetColor(
                0.2 + 0.8 * danger, 
                0.6 * (1 - danger), 
                1.0 * (1 - danger)
            )
        else:
            self.robot_actor.GetProperty().SetColor(0.2, 0.6, 1.0)
        
        # 更新信息文本
        status = "⚠ 避障中!" if dist_to_obs < safe_dist * 1.2 else "✓ 安全"
        self.info_text.SetInput(
            f"时间: {self.t:.1f}s\n"
            f"目标方向: {direction}\n"
            f"机器人: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})\n"
            f"目标点: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})\n"
            f"到目标距离: {dist_to_goal:.2f}m\n"
            f"到障碍物距离: {dist_to_obs:.2f}m (安全:{safe_dist:.2f}m)\n"
            f"状态: {status}"
        )
        
        # 更新时间
        self.t += self.dt
        
        # 渲染
        self.window.Render()
    
    def _on_key(self, obj, event):
        """键盘回调"""
        key = self.interactor.GetKeySym().lower()
        
        if key == 'space':
            self.paused = not self.paused
            state = "暂停" if self.paused else "播放"
            print(f"[{state}]")
        elif key == 'r':
            self._reset()
            print("[重置]")
        elif key == 'q':
            print("[退出]")
            self.interactor.TerminateApp()
    
    def _reset(self):
        """重置仿真"""
        self.t = 0.0
        self.controller.reset()
        self.robot_history.clear()
        self.goal_history.clear()
        self.obs_history.clear()
        
        # 重置位置
        self.robot_actor.SetPosition(self.cfg.START_POINT)
        goal = self.controller.get_target_position(0)
        obs = self.controller.obstacle_position(0)
        self.goal_actor.SetPosition(goal)
        self.obstacle_actor.SetPosition(obs)
        self.safety_actor.SetPosition(obs)
        
        self.window.Render()
    
    def run(self):
        """运行可视化"""
        cfg = self.cfg
        print("=" * 60)
        print("3D MPC 轨迹跟踪与动态避障可视化")
        print("=" * 60)
        print(f"\n配置参数:")
        print(f"  起点: {cfg.START_POINT}")
        print(f"  终点: {cfg.END_POINT}")
        print(f"  目标速度: {cfg.TARGET_SPEED} m/s")
        print(f"  障碍物半径: {cfg.OBSTACLE_RADIUS} m")
        print(f"  安全边距: {cfg.SAFETY_MARGIN} m")
        print(f"\n控制说明:")
        print("  空格键  - 暂停/播放")
        print("  R      - 重置仿真")
        print("  Q      - 退出程序")
        print("  鼠标左键拖动 - 旋转视角")
        print("  鼠标滚轮    - 缩放")
        print("  鼠标中键拖动 - 平移视角")
        print("\n启动中...\n")
        
        self.window.Render()
        
        # 创建定时器 (~30 FPS)
        self.interactor.CreateRepeatingTimer(33)
        
        # 启动交互
        self.interactor.Start()


if __name__ == "__main__":
    # ====== 在这里修改配置参数 ======
    
    # 目标运动设置
    # Config.START_POINT = np.array([0, 0, 0])    # 起点
    # Config.END_POINT = np.array([10, 8, 6])     # 终点
    # Config.TARGET_SPEED = 1.5                    # 速度 (m/s)
    
    # 轨迹长度设置 (点数, 每点间隔0.1秒)
    # Config.ROBOT_TRAIL_LENGTH = 800              # 机器人轨迹 (800点=80秒)
    # Config.GOAL_TRAIL_LENGTH = 600               # 目标轨迹 (600点=60秒)
    # Config.OBS_TRAIL_LENGTH = 400                # 障碍物轨迹 (400点=40秒)
    
    # 障碍物设置
    # Config.OBSTACLE_CENTER = np.array([4, 3, 2.5])   # 运动中心
    # Config.OBSTACLE_AMPLITUDE = np.array([2, 2, 1.5]) # 运动幅度
    # Config.OBSTACLE_FREQ = np.array([0.25, 0.3, 0.35]) # 运动频率
    # Config.OBSTACLE_RADIUS = 1.0                  # 半径
    
    viz = VTKVisualizer()
    viz.run()