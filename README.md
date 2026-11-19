# RGB-D 多帧配准与三维实例地图

学习与工程实践项目：从 RGB-D 序列构建带稳定实例 ID 的三维物体地图，并支持简单文本检索。

第一关和第二关的基础实现与实验检查已完成，包括刚体拟合、点到点 ICP、真实帧对照、短序列累计、采样对照及换起点迁移。实验记录保留了 ICP 的对照结果和累积漂移较大的区间。第三关已通过 ARKitScenes 数据适配、逐帧点云导出、跨帧关联与命令行复现验收。当前进入第四关，从地图实例回查二维观测，为多视角特征提取准备输入。

![第一关：已知位姿融合地图与所选帧的相机中心轨迹](assets/stage01_map_trajectory.png)

图中红色折线连接所选帧的参考相机中心；地图与轨迹使用相同世界坐标系。此图展示提供位姿下的融合，不代表独立估计的轨迹。

## 目标流程

RGB-D 与相机参数 → 时间关联与反投影 → 多帧几何对齐 → 二维实例提取 → 跨帧实例关联与融合 → 三维物体地图 → 文本检索。

二维前端计划复用 Grounding DINO + SAM 2，图文特征计划使用 SigLIP 2。核心练习是几何、配准、跨帧关联、融合与评估。

## 学习与实现

- [五关路线及验收](docs/ROADMAP.md)
- [第一关：已知位姿下的多帧融合](docs/STAGE_01.md)
- [第一关实测与检查记录](reports/stage01_validation.md)
- [第二关：对应点刚体拟合与 ICP](docs/STAGE_02.md)
- [对应点刚体拟合实测记录](reports/stage02_rigid_fit.md)
- [点到点 ICP 实现任务](docs/STAGE_02_ICP.md)
- [合成点云 ICP 实测记录](reports/stage02_icp_synthetic.md)
- [实现说明：真实 RGB-D 帧对配准](docs/STAGE_02_RGBD.md)
- [真实帧对实测与迭代上限诊断](reports/stage02_rgbd_pair.md)
- [短序列累计与同区间采样对照](reports/stage02_sequence.md)
- [第三关：ARKitScenes 适配与三维实例地图](docs/STAGE_03.md)
- [第三关单场景数据检查](reports/stage03_data_check.md)
- [第三关 A：新数据适配与独立区间重建](reports/stage03_arkit_preview.md)
- [第三关 B：逐帧实例导出实测与分割失败案例](reports/stage03_instance_inputs.md)
- [第三关 C：跨帧关联与体素重叠对照](docs/STAGE_03.md#c-跨帧关联与实例融合)
- [第四关：多视角特征与文本检索](docs/STAGE_04.md)

核心算法由学习者实现；可以调用图像读写、四元数转换、KD-tree、SVD 和点云显示等底层工具。助手协助解释、审查、调试、命令行入口整理和展示材料。

## 目录

```text
rgbd_instance_map/
├── README.md
├── requirements-geometry.txt
├── .gitignore
├── tum_io.py             # TUM 读取与时间关联
├── geometry.py           # 位姿、变换与插值
├── registration.py       # 刚体拟合、ICP 与评价
├── fuse_known_poses.py   # 已知位姿融合入口
├── register_rgbd_pair.py # 帧对配准命令行入口
├── register_rgbd_sequence.py # 相邻帧配准与累计拼接
├── arkit_io.py           # ARKitScenes 读取与帧关联
├── preview_arkit.py      # 提供位姿下的场景预览
├── extract_instances.py  # 掩码到世界坐标实例观测
├── instance_map.py       # 同类实例的跨帧关联与点云融合
├── docs/
│   ├── ROADMAP.md
│   ├── STAGE_01.md
│   ├── STAGE_02.md
│   ├── STAGE_02_ICP.md
│   ├── STAGE_02_RGBD.md
│   └── STAGE_03.md
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

## 第二关：帧对配准运行

两种 ICP 都从单位矩阵开始，估计 source 相机坐标到 target 相机坐标的变换。参考位姿在 RGB 时间上插值，只用于误差评价。以下命令从项目根目录运行：

```powershell
python register_rgbd_pair.py --sequence data/tum/rgbd_dataset_freiburg1_desk --source-index 5 --target-index 0 --max-iterations 50 --output outputs/stage02_pair5_iter50
```

`--source-index` 和 `--target-index` 指关联成功列表中的位置，不是原始 RGB 行号。将 source 改为 10 或 20 可重复帧距对照。第 20 组增加迭代上限的诊断另存到新目录：

```powershell
python register_rgbd_pair.py --sequence data/tum/rgbd_dataset_freiburg1_desk --source-index 20 --target-index 0 --max-iterations 200 --output outputs/stage02_pair20_iter200
```

可选参数：`--voxel-size 0.02`（米）、`--max-distance 0.10`（米）、`--tolerance 1e-6`（相邻记录 RMSE 差，米）。`--max-iterations` 同时控制自写与 Open3D 的轮数上限。输入反投影沿用当前 TUM 协议：默认 ROS 内参、深度除以 5000、保留 `0 < Z < 4` 米；更换相机或数据集前需确认这些设置。

增加 `--show` 会在保存后依次显示初始状态、自写 ICP 和 Open3D ICP；关闭当前窗口后进入下一组。不带该参数时保存后直接退出。`python register_rgbd_pair.py --help` 查看全部参数。

每次输出：

```text
指定输出目录/
├── metrics.csv             # 初值、自写、Open3D 的几何和位姿指标，含输入与主要参数
├── icp_history.csv         # 自写 ICP 每轮更新前的 RMSE，单位米
├── T_own.npy               # 自写 ICP 的 source→target 4×4 矩阵
├── T_open3d.npy             # Open3D 的 source→target 矩阵
├── T_ref.npy               # 只用于评价的参考相对变换
├── target.ply              # target 相机坐标点云
├── aligned_own_icp.ply      # 自写 ICP 变换后的 source 点云
└── aligned_open3d_icp.ply   # Open3D 变换后的 source 点云
```

三个 PLY 均在 target 相机坐标系，保留 RGB 颜色。终端中的 RMSE、平移误差显示为毫米；CSV 使用 `inlier_rmse_m`、`translation_error_m` 和 `rotation_error_deg` 明确单位，fitness 范围为 0～1。矩阵可通过 `np.load("输出目录/T_own.npy")` 读取。

自写 ICP 的 `num_updates` 与 `stop_reason` 来自实际 history；Open3D 当前接口不返回实际轮数，这两列留空。`rmse_tolerance` 只表示满足误差变化门槛，不保证位姿准确。最终评价重新计算返回矩阵的分数，不直接拿 history 的最后一项代替。同一输出目录重跑覆盖同名文件，保留对照实验时使用不同目录。

## 第二关：短序列累计运行

```powershell
python register_rgbd_sequence.py --sequence data/tum/rgbd_dataset_freiburg1_desk --stride 1 --max-frames 96 --max-iterations 200 --output outputs/sequence_stride1
```

输出 `estimated_map.ply`、`reference_map.ply` 与 `trajectory_errors.csv`。两张地图均位于所选首帧相机坐标系，估计地图使用相邻 ICP 结果累计；参考位姿只参与评价与参考地图生成。增加 `--show` 可在保存后依次打开两张地图。

当前 CSV 记录所选序列帧索引、累计旋转误差（度）与累计平移误差（米）。同区间采样对照与精度边界见 [短序列报告](reports/stage02_sequence.md)；保存点云为各帧下采样点云直接叠加。

通过 `--start-index` 切换所选首帧，已检查的迁移区间为关联索引 50～145：

```powershell
python register_rgbd_sequence.py --sequence data/tum/rgbd_dataset_freiburg1_desk --start-index 50 --stride 1 --max-frames 96 --max-iterations 200 --output outputs/sequence_start50_stride1
```

![短序列采样对照与换起点的误差曲线](assets/stage02_sequence_errors.png)

首段相同区间内，步长 1 的末帧误差为 4.55°、175.43 mm；换起点区间为 43.94°、983.39 mm，显示当前局部 ICP 前端的累积漂移。右列是不同时间区间的迁移检查，不作为采样参数对照。完整协议和核对方法见报告。

## 第三关：跨帧实例地图运行

ARKitScenes 场景 `47333462` 的逐帧三维观测已导出。三帧椅子的有效深度像素为 12898、10470、11964，按 0.02 米体素下采样后分别为 1671、1789、1923 点。床样例迁移得到 1658、814、2261 点，几何与 JSON 核对通过。床的掩码覆盖部位差异明显，作为分割失败案例保留。[第三关 B 实测](reports/stage03_instance_inputs.md)

`extract_instances.py` 当前通过脚本开头的 `mask_dir` 和 `output_dir` 选择输入输出，当前配置为床样例。椅子缓存位于 `outputs/stage03_chair_views_masks/`，已导出的观测位于 `outputs/stage03_observations/`。当前脚本逐目标打开点云窗口，关闭后继续处理，全部目标处理完后保存 `observations.json`。入口参数化属于后续工程整理任务。

第三关 C 在 `instance_map.py` 实现了同类候选筛选、质心距离与体素 IoU 筛选、同帧一对一匹配和体素融合。真实三帧椅子合成一个地图 ID，合成双椅交换检测编号后保留两个地图 ID。错合并对照中，质心距离约 0.406 米、体素 IoU 约 0.037。只用距离得到一个实例，加 IoU 门槛后保留两个实例。命令行入口已复现这组对照，换回原始三帧椅子时得到一个实例、三条来源和 2621 个融合点。[第三关 C 实测与复现命令](docs/STAGE_03.md#6-体素重叠对照实测)

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
