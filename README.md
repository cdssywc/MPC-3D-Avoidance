# MPC-3D-Avoidance
Real-time 3D trajectory tracking and obstacle avoidance using Model Predictive Control (MPC) with GPU acceleration
这是一个基于模型预测控制（MPC） 的3D轨迹跟踪与避障演示程序。程序实现了在动态障碍物环境中的自主导航，集成了卡尔曼滤波器进行障碍物轨迹预测，支持GPU加速，并提供了实时的3D可视化界面。
版本说明
mpc_0.py无卡尔曼滤波处理
mpc_1.py有卡尔曼滤波处理，预测障碍物的下一时刻运动轨迹
mpc_2.py有卡尔曼滤波处理，预测障碍物的下一时刻运动轨迹，加入qt5参数设置
<img width="1282" height="815" alt="f672a47c32d03519faa39b0892e2497" src="https://github.com/user-attachments/assets/cdc267f4-23f2-4e13-ae4d-c243c0f0def8" />
<img width="1272" height="880" alt="8e8121afb130627dc5b3e6f0283b822" src="https://github.com/user-attachments/assets/1b643eaf-0d0f-4bc3-8230-f0206125f646" />
<img width="1886" height="1106" alt="5973ed33dd9982b5848a2562046daa1" src="https://github.com/user-attachments/assets/17509fa0-d079-4695-83f4-65cd1d2c97bd" />
<img width="302" height="375" alt="34713336c4903aa6002e4d15d6020bd" src="https://github.com/user-attachments/assets/3e8a8512-a900-43fa-bce9-857eb2e66554" />
<img width="809" height="307" alt="d2beab24a2c133c3d98091696990bdc" src="https://github.com/user-attachments/assets/d7856589-d434-416b-8bc2-206d04115881" />
<img width="497" height="226" alt="416d9d1a4713aba0f8d4b120dedf36d" src="https://github.com/user-attachments/assets/28f97449-21e9-40ed-8bf4-b5c905b87310" />
<img width="725" height="517" alt="b9678639c72354313f6a33c6a29ca78" src="https://github.com/user-attachments/assets/66500bd9-976f-4a2c-9c46-b199eaa5d26d" />

核心特性
🎯 MPC控制器 - 实时轨迹跟踪与避障优化

🎯 卡尔曼滤波 - 障碍物运动状态估计与轨迹预测

⚡ GPU加速 - 自动检测CUDA/MPS/CPU并优化

🎨 3D可视化 - 基于VTK的实时渲染界面

📊 多线程架构 - 计算与渲染分离，视图旋转不影响仿真

🔀 随机障碍物 - 可变速度与方向的动态障碍物

📈 性能监控 - 实时显示距离、速度、加速度等信息

📸 演示效果
<div align="center">
主界面	避障演示
<img src="https://via.placeholder.com/400x250/333333/FFFFFF?text=3D+Visualization+Interface" width="400">	<img src="https://via.placeholder.com/400x250/333333/FFFFFF?text=Obstacle+Avoidance+Demo" width="400">
</div>
界面元素说明
🔵 蓝色小球 - 机器人（受控主体）

🔴 红色小球 - 随机运动障碍物

🟢 绿色小球 - 动态目标点

🟡 黄色点 - 卡尔曼滤波器预测的障碍物未来轨迹

⚪ 透明红色球 - 安全边界（机器人需要避开此区域）

🔷 蓝色轨迹线 - 机器人历史运动路径

🟥 红色轨迹线 - 障碍物历史运动路径

🟩 绿色轨迹线 - 目标点历史位置

🚀 快速开始
系统要求
操作系统: Windows 10+, Ubuntu 18.04+, macOS 10.15+

Python: 3.8 或更高版本

可选GPU: NVIDIA GPU（支持CUDA）或Apple Silicon（M1/M2/M3）

安装步骤
方法1: 使用pip（推荐）
bash
# 克隆项目
git clone https://github.com/cdssywc/mpc-3d-avoidance.git
cd mpc-3d-avoidance

# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
方法2: 使用conda
bash
# 使用conda创建环境
conda env create -f environment.yml
conda activate mpc-3d-avoidance
方法3: 手动安装
bash
# 基本依赖
pip install numpy scipy vtk

# 可选GPU支持（根据需要选择）
pip install torch  # PyTorch（自动检测CUDA/MPS）
运行程序
bash
python mpc_1.py
程序启动后将显示：

硬件检测信息（GPU/CPU模式）

配置参数摘要

3D可视化窗口

🎮 交互控制
键盘控制
按键	功能	说明
Space	暂停/恢复仿真	暂停时仍可旋转视角
R	重置仿真	恢复初始状态
Q	退出程序	安全关闭所有线程
F	全屏切换	切换全屏模式
S	截图	保存当前视图为PNG
鼠标控制
操作	功能
左键拖拽	旋转视角
右键拖拽	平移视角
滚轮滚动	缩放视图
中键拖拽	快速缩放
界面区域说明
text
┌─────────────────────────────────────────────────────────────┐
│  标题栏: 显示程序名称和GPU/CPU模式                         │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │                3D可视化区域                        │   │
│  │  • 坐标系原点: (0,0,0)                             │   │
│  │  • 网格地面: z=0平面                               │   │
│  │  • 坐标轴: 红(X)、绿(Y)、蓝(Z)                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  信息面板 (左下)                                   │   │
│  │  • 时间、位置、距离信息                            │   │
│  │  • 卡尔曼滤波器状态                               │   │
│  │  • 安全状态指示器                                 │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  图例面板 (右下)                                   │   │
│  │  • 颜色图例说明                                   │   │
│  │  • 控制快捷键列表                                 │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
⚙️ 配置参数详解
配置文件结构
所有参数在 Config 类中定义，可在 main() 函数中修改。

核心参数说明
1. 目标运动参数
python
cfg.START_POINT = np.array([0.0, 0.0, 0.0])    # 起始点坐标
cfg.END_POINT = np.array([10.0, 8.0, 6.0])     # 终点坐标
cfg.TARGET_SPEED = 1.2                          # 目标移动速度 (m/s)
目标点在起始点和终点之间来回移动

移动距离 = 2 * ‖END_POINT - START_POINT‖

2. 障碍物参数
python
cfg.OBSTACLE_INITIAL_POS = np.array([5.0, 4.0, 3.0])  # 初始位置
cfg.OBSTACLE_SPEED_RANGE = (0.8, 2.5)                  # 速度范围 (m/s)
cfg.OBSTACLE_SPEED_CHANGE_RATE = 0.5                   # 速度变化率
cfg.OBSTACLE_DIRECTION_CHANGE_RATE = 0.15              # 方向变化率
cfg.OBSTACLE_RADIUS = 0.8                              # 障碍物半径 (m)
cfg.SAFETY_MARGIN = 0.5                                # 安全边界厚度 (m)
cfg.OBSTACLE_BOUNDS = ((-1, 4), (-1, 4), (0, 4))       # 运动范围
安全距离 = 障碍物半径 + 安全边界

机器人必须保持在安全距离之外

3. 机器人参数
python
cfg.ROBOT_RADIUS = 0.35          # 机器人半径 (m)
cfg.MAX_VELOCITY = 3.0           # 最大速度 (m/s)
cfg.MAX_ACCELERATION = 2.5       # 最大加速度 (m/s²)
4. MPC控制器参数
python
cfg.HORIZON = 15                 # 预测时域（步数）
cfg.CONTROL_HORIZON = 8          # 控制时域（步数）
cfg.MPC_ITERATIONS = 30          # 优化迭代次数
预测时域: MPC向前预测的时间步数

控制时域: 优化控制序列的长度（≤预测时域）

迭代次数: 优化算法的最大迭代次数

5. 卡尔曼滤波器参数
python
cfg.KALMAN_PROCESS_NOISE = 0.8   # 过程噪声协方差
cfg.KALMAN_MEASUREMENT_NOISE = 0.1  # 测量噪声协方差
6. 可视化参数
python
cfg.ROBOT_TRAIL_LENGTH = 60      # 机器人轨迹长度
cfg.GOAL_TRAIL_LENGTH = 60       # 目标点轨迹长度
cfg.OBS_TRAIL_LENGTH = 40        # 障碍物轨迹长度
cfg.PREDICTION_TRAIL_LENGTH = 15 # 预测轨迹长度
cfg.WINDOW_SIZE = (1500, 950)    # 窗口尺寸
📊 性能优化指南
GPU加速模式
程序自动检测并选择最佳计算设备：

python
# 优先级顺序
1. CUDA (NVIDIA GPU) - 最快
2. MPS (Apple Silicon) - Apple芯片优化
3. CPU + PyTorch - CPU上的张量优化
4. CPU + NumPy - 纯NumPy后备方案
检测结果示例:

text
[GPU] CUDA detected: NVIDIA GeForce RTX 3080
[GPU] Apple MPS detected
[CPU] No GPU detected, using CPU with PyTorch optimization
[CPU] PyTorch not installed, using NumPy only
多线程架构
text
┌─────────────────────────────────────────────────────┐
│                   主线程                            │
│  • 启动/停止仿真                                   │
│  • 处理用户输入                                    │
│  • 管理子线程                                      │
├─────────────────────────────────────────────────────┤
│              计算线程 (Compute Thread)              │
│  • MPC优化计算                                     │
│  • 卡尔曼滤波更新                                  │
│  • 状态更新 (100Hz)                                │
│  • 与渲染线程异步                                  │
├─────────────────────────────────────────────────────┤
│              渲染线程 (Render Thread)              │
│  • 3D图形渲染                                      │
│  • 界面更新 (30FPS)                                │
│  • 独立于计算线程                                  │
└─────────────────────────────────────────────────────┘
性能监控
程序实时显示以下性能指标：

计算时间: MPC优化耗时

帧率: 渲染帧率

内存使用: GPU/CPU内存

距离指标: 与障碍物、目标的距离


权重设置 (可在代码中调整):

python
self.w_goal = 1.0      # 目标跟踪权重
self.w_control = 0.1   # 控制量权重
self.w_obs = 3000.0    # 避障权重 (较高)
self.w_smooth = 0.05   # 平滑性权重
