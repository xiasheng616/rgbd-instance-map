# RGB-D 多帧配准与三维实例地图

学习与工程实践项目：从 RGB-D 序列构建带稳定实例 ID 的三维物体地图，并支持简单文本检索。

当前状态：第一关已通过验收，完成时间关联、提供位姿下的多帧彩色融合、相机轨迹显示与命令行保存；准备进入第二关的刚体配准与 ICP。尚无自行估计位姿、实例关联或检索的实测结果。

![第一关：已知位姿融合地图与所选帧的相机中心轨迹](assets/stage01_map_trajectory.png)

图中红色折线连接所选帧的参考相机中心；地图与轨迹使用相同世界坐标系。此图展示提供位姿下的融合，不代表独立估计的轨迹。

## 目标流程

RGB-D 与相机参数 → 时间关联与反投影 → 多帧几何对齐 → 二维实例提取 → 跨帧实例关联与融合 → 三维物体地图 → 文本检索。

二维前端计划复用 Grounding DINO + SAM 2，图文特征计划使用 SigLIP 2。核心练习是几何、配准、跨帧关联、融合与评估。

## 学习与实现

- [五关路线及验收](docs/ROADMAP.md)
- [第一关：已知位姿下的多帧融合](docs/STAGE_01.md)
- [第一关实测与检查记录](reports/stage01_validation.md)

核心算法由学习者实现；可以调用图像读写、四元数转换、KD-tree、SVD 和点云显示等底层工具。助手协助解释、审查、调试和整理展示材料。

## 目录

```text
rgbd_instance_map/
├── README.md
├── requirements-geometry.txt
├── .gitignore
├── docs/
│   ├── ROADMAP.md
│   └── STAGE_01.md
├── data/                 # 本地数据，不提交 Git
├── outputs/              # 完整输出，不提交 Git
├── assets/               # 后续放精选截图和演示
└── reports/              # 后续放小型实测结果和分析
```

第一关已包含 `tum_io.py`（读取与时间关联）、`geometry.py`（位姿矩阵与点变换）、`fuse_known_poses.py`（融合运行入口）。后续按实际复杂度整理模块。

## 第一关运行

先按第一关任务说明准备完整的 TUM `freiburg1_desk` 序列，然后从项目根目录运行：

```powershell
python fuse_known_poses.py --sequence data/tum/rgbd_dataset_freiburg1_desk --stride 20 --max-frames 30 --voxel-size 0.02 --output outputs/stage01_stride20
```

增加 `--show` 可打开地图与红色相机轨迹的显示窗口；不带此参数时保存后退出。采样步长针对关联成功的记录。输出为指定目录下的 `fused_map.ply`，不包含轨迹线；轨迹目前在运行时构造并显示。

## 基础环境

第一关沿用已经用于点云实验的 `yolov11` 环境即可；新项目代码不依赖 YOLO。基础库版本在 2026-11-08 从该环境的安装元数据读取，尚未验证新建环境完整安装。

```powershell
conda activate yolov11
Set-Location E:\shixi\rgbd_instance_map
python -c "import numpy, scipy, cv2, open3d; print('geometry imports OK')"
```

`requirements-geometry.txt` 记录几何阶段的库版本。模型阶段再确定 PyTorch、Transformers、Grounding DINO 和 SAM 2 的兼容环境，届时补充完整安装说明。

## 工程与发布约定

- 输入目录、采样间隔、处理帧数和输出目录通过命令行传入，避免写死个人绝对路径。
- 每个阶段保留一个可运行入口、一份小型实测报告和代表性结果；可复用逻辑写成函数。
- 数据、模型权重、完整缓存和完整地图不上传 Git；README 链接到获取方式。
- 只展示实际完成的能力和测量结果。阶段通过后，由学习者亲自操作 Git 提交及最终 GitHub 上传；助手讲解操作步骤、检查提交内容和远端结果，不代为提交或上传本项目。
- 若引用或改编第三方代码，逐项记录来源与许可；尚未决定本项目自有代码许可。

## 能力边界

第一关使用提供位姿做重建，不是独立 SLAM。后续 ICP 前端也不自动构成完整 SLAM。第一版以静态室内、离线建图、类别与简单属性检索为范围；物体点云代表已观测表面，不代表完整模型或真实质心。
