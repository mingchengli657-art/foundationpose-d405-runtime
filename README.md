# FoundationPose D405 Runtime

**Piper 视觉抓取系列 · 03 / 连续位姿跟踪** · [系列总入口与整套运行指南](https://github.com/mingchengli657-art/piper-vision-grasping)

本模块使用[物体建模](https://github.com/mingchengli657-art/d405-object-modeling)导出的模型；位姿可经[手眼标定](https://github.com/mingchengli657-art/piper_handeye_calibration)结果转换后交给 [Piper 控制](https://github.com/mingchengli657-art/piper-known-object-control)。

将 Intel RealSense D405 的 RGB-D 输入接入 FoundationPose，持续绘制已知物体的三维框和坐标轴，并发布相机坐标系下的 6D 位姿。由机器人比赛工程中的 `Piper_Control/Foundation Pose` 提取。

本项目的工作是 **D405 接入、ROS 与 GPU 环境分离、帧队列、模型检查、深度一致性筛选、跟踪丢失状态以及对称物体姿态稳定**。FoundationPose 网络和推理算法来自 [NVIDIA NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)，随包 `upstream/` 保留了本地使用的源码快照及许可证。

这是模型驱动的单物体位姿跟踪：需要该物体的米制 OBJ，首次由操作者圈定物体。它不能直接对任意画面自动分类所有物体。正常运行不再需要 ChArUco 板，也不需要机械臂。

## 流程与接口

```text
D405 彩色 / 对齐深度 / 内参
          ↓
ROS I/O（系统 Python / ROS 2）
          ↓  本机文件队列 runtime/frames
GPU worker（FoundationPose 环境）
          ↓  位姿 / 深度检查 / 框与坐标轴
ROS I/O → 相机位姿及模型标识 → 外部控制程序
```

| 输入 / 输出 | 默认接口 |
|---|---|
| 彩色图 | `/camera/d405/color/image_raw` |
| 对齐彩色的深度图 | `/camera/d405/aligned_depth_to_color/image_raw` |
| 内参 | `/camera/d405/color/camera_info` |
| 相机坐标位姿 | `/foundationpose/object_pose_camera`，`geometry_msgs/msg/PoseStamped` |
| 模型标识和完整矩阵 | `/foundationpose/object_pose_json`，`std_msgs/msg/String` |
| 跟踪恢复状态 | `/foundationpose/recovery_required`，`std_msgs/msg/Bool` |

`camera_T_object` 将模型坐标系中的点变换到彩色相机光学坐标系；单位为米、四元数 `[x,y,z,w]`，时间戳沿用输入图像。物体模型原点和轴由建模包定义，不等同于抓取点。

## 下载后先做离线检查

在仓库根目录创建普通 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -v
bash scripts/validate_model.sh examples/model_005_final
```

附带一个真实瓶子凸包模型，不附带采集图像或神经网络权重。此步骤验证模型接口、网格、深度门限、帧队列及姿态稳定函数，不运行 GPU 推理。测试环境和 GPU 推理环境应分开；测试依赖不是完整的 FoundationPose 安装清单。

## 准备 GPU 推理环境

需要 NVIDIA GPU、兼容的驱动、CUDA 工具链和 FoundationPose 的 Python 依赖。按随包 `upstream/readme.md` 与 [上游安装说明](https://github.com/NVlabs/FoundationPose#env-setup-option-2-conda-local)准备；CUDA / PyTorch / PyTorch3D 的组合取决于实际显卡，不能把一台机器的编译产物直接搬到另一台。

```bash
conda env create -f upstream/environment.yml
conda activate foundationpose
# 按上游说明安装与你的 CUDA 匹配的 PyTorch、PyTorch3D 和 nvdiffrast
python -m pip install -r upstream/requirements.txt
bash scripts/build_mycpp.sh
```

`upstream/environment.yml` 的 worker 基线为 Python 3.11。原源码的 mycpp C++ 扩展必须在目标环境重新编译；本仓库没有打包 `.so`、Conda 环境或 GPU 库。模型驱动推理不要求启动 BundleSDF 的模型自由建模流程。

从 [上游权重下载入口](https://github.com/NVlabs/FoundationPose#data-prepare) 获取 refiner 与 scorer，目录必须如下：

```text
upstream/weights/
├── 2023-10-28-18-33-37/
│   ├── config.yml
│   └── model_best.pth
└── 2024-01-11-20-02-45/
    ├── config.yml
    └── model_best.pth
```

权重约 247 MB，已从源码上传包排除。如果你的机器仍保留原工程，可以手动复制原 `Piper_Control/Foundation Pose/upstream/weights/` 到这里，无需重复下载。本地快照保留了原先限制 checkpoint 加载的补丁，详见 `docs/PROVENANCE.md`。

## 启动 D405 与 ROS 输入

先安装、启动 D405 ROS 2 驱动；相机驱动不包含在本项目中。彩色和深度必须已对齐，只有分辨率相同并不足够。浮点深度按米解释，整数深度默认 `--depth-scale 0.001`。

使用独立终端和系统 Python，避免 GPU 环境污染 ROS 的 Python / NumPy：

```bash
source /opt/ros/humble/setup.bash
# ROS 环境需要 cv_bridge、message_filters、NumPy、SciPy、PyYAML
bash scripts/start_ros_io.sh
```

可用 `--color-topic`、`--depth-topic`、`--camera-info-topic` 指定你的话题，或用 `ROS_SETUP` 指定 ROS 安装路径。脚本不自动读取个人比赛依赖文件；确有需要时显式设置 `FP_DEPENDENCIES_SH`。

## 启动连续跟踪

第二个终端激活准备好的 FoundationPose 环境，从本仓库运行：

```bash
conda activate foundationpose
export FP_PYTHON=python
export CUDA_HOME=/usr/local/cuda  # 改为本机实际位置
bash scripts/start_worker.sh examples/model_005_final
```

替换示例目录即可使用 `d405-object-modeling` 导出的任意合格模型包。worker 首先检查米制、网格闭合、面绕序和体积，再要求你画首次掩膜：左键添加点、右键撤销、Enter 确认，圈紧目标物体。首次注册时保持物体和相机稳定。后续窗口持续更新框和轴；`R` 重新圈定、`Q` / Esc 退出。

worker 与 ROS I/O 必须在同一机器访问同一 IPC 目录，默认本仓库 `runtime/`；如果修改，两个终端都设置相同的 `FP_IPC_DIR`。先启动 ROS I/O，再启动 worker，且每个 IPC 目录只运行一对进程。ROS I/O 启动时清理旧帧与结果，运行中重启它时应同时重启 worker。

默认最多输入 30 Hz，FIFO 容量两帧，实际推理频率由 GPU 和队列决定，不能理解为保证 30 FPS。`FP_CAPTURE_HZ`、`FP_QUEUE_SIZE` 控制输入；`FP_OVERLAY_SAVE_HZ` 默认 2 Hz，只控制 `latest_overlay.png` 的磁盘保存频率，实时窗口仍按处理帧更新。

## 模型包和对称性

模型目录至少包含 `model.yaml` 和其中相对 `mesh` 路径指向的 OBJ：

```yaml
model_id: my_object
units: meters
model_frame: object_model_center
mesh: object_foundationpose.obj
```

附带瓶子模型额外开启绕模型 +Z 的轴向旋转稳定。这消除了圆柱外形难以观测的轴向漂移，不是恢复了物体的真实轴向角。非对称物体应移除该配置，不要随意把所有物体锁轴。

渲染深度与观测深度不符或位姿跳变时，结果被拒绝；连续拒绝进入恢复状态，连续合格帧后解除。恢复话题当前是状态变化事件，后加入的订阅者可能错过先前事件，控制侧还必须检查位姿新鲜度。显示框存在不等于允许执行抓取。

## 与机械臂代码连接

控制侧另行加载实际手眼外参和 TCP 反馈：

```text
base_T_object = base_T_tcp × tcp_T_camera × camera_T_object
```

本项目只提供相机位姿，不订阅机械臂反馈或发布控制指令。控制模块的 `handeye_pose_bridge.py` 才负责基座坐标转换。

## 目录与发布

```text
app/           个人项目的相机、ROS、IPC、跟踪与检查代码
scripts/       启动和网格验证脚本
upstream/      NVIDIA FoundationPose 本地源码快照及许可证
examples/      一个轻量米制真实模型
tests/         不依赖 ROS / CUDA 的检查
docs/          来源、变更、上游版本和已知限制
```

可把目录作为独立仓库上传。`.gitignore` 排除 `runtime/`、权重、缓存和编译产物；GitHub Actions 只运行 CPU 侧测试，不代表 CUDA、相机或整套跟踪已实测通过。

请保留上游版权与 `upstream/LICENSE`。其中 NVIDIA Source Code License 有研究/评估用途限制；本次没有将 NVIDIA 算法声明为个人原创，也没有替你给全部代码统一添加 MIT 等许可证。
