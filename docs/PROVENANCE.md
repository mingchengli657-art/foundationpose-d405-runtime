# 来源与整理记录

本包从比赛工程 `Piper_Control/Foundation Pose/` 提取。原运行说明保留为 `ORIGINAL_README.md`，以新 README 为整理版使用入口。

## 上游算法

- 来源：https://github.com/NVlabs/FoundationPose
- 本地上游仓库基准提交：`a1b694b83e633c2cb6115b9063d940a687759392`。
- `upstream/` 是比赛实际使用的本地源码快照，并非保证与该提交逐字一致的干净 checkout。
- 已保留的本地修改：`learning/training/predict_score.py` 和 `predict_pose_refine.py` 使用 `torch.load(..., map_location='cpu', weights_only=True)`。这需要支持该参数的 PyTorch；未知 checkpoint 格式仍需实际验证。
- 原始版权和 NVIDIA Source Code License 保留在 `upstream/LICENSE`。上游代码、网络和推理算法不属于个人原创。
- 权重、编译产物、缓存、相机采集数据和个人环境未随包发布。

## 项目接入代码

`app/` 和启动脚本保留比赛工程的 ROS 输入、GPU worker、首次圈选、持续跟踪、文件队列、米制模型验证、深度筛选和轴向稳定逻辑。本次没有重新训练网络。

本次整理增加 CPU 回归测试、CI、可移植路径、轻量模型示例与新 README。损坏的 NPZ 帧会隔离为 `.invalid`，避免持续占满队列。环境脚本仅在显式设置时读取额外依赖文件。

## 模型示例

`examples/model_005_final/` 来自原 `object_modeling/datasets/object005/model005final/`，仅保留模型描述和米制 OBJ。删除了指向未随包提供的点云字段。该模型是建模流程导出的瓶子凸包，供接口验证，不是标准评测数据集。

## 验证边界

整理时通过 8 项 CPU 测试和示例网格验证。此次没有连接 D405 或验证 CUDA 实时推理；GitHub 绿色检查仅覆盖 CPU 模块。安装和运行仍需分别验证 GPU、相机、ROS 时间戳和模型匹配。
