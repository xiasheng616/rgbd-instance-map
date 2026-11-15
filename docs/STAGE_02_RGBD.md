# 第二关 C：真实 RGB-D 帧对与配准对照

启动日期：2026-11-13。更新日期：2026-11-15。参考位姿插值、误差评价、真实帧对对照、短序列累计、采样对照及换起点的基础实现与实验检查已完成。见 [帧对实测](../reports/stage02_rgbd_pair.md)、[短序列与失败区间](../reports/stage02_sequence.md)。当前 ICP 基线存在累积漂移，材料保留该局限。运行方法见 [README](../README.md#第二关帧对配准运行)。下文保留任务提出时的实现过程，第二关 Git 发布仍由学习者操作。

## 本关目标与实现过程

先完成一个真实两帧配准入口：以同一输入、初值和阈值对比自己的 ICP 与 Open3D，再使用正确配时的参考相对位姿评估。两帧流程稳定后扩展到短序列与失败分析，届时才完成第二关。

## 现在先写的完整任务

新建 `register_rgbd_pair.py`，复用 TUM 读取、时间关联、反投影与现有 ICP。

### 1. 选择帧与坐标约定

初次使用关联结果 `matches[0]` 作为 target，`matches[5]` 作为 source。索引指关联列表的位置，不是原始 RGB 行号。

分别读取正确配对的 RGB/depth，产生各自相机坐标系的点云。沿用已验收设置：TUM PNG 深度除以 5000，`0 < Z < 4.0` 米，ROS 默认内参，RGB 颜色，0.02 米体素下采样。

将单帧点云生成部分抽成函数，避免复制两份反投影。该函数返回相机坐标点云；不在其中使用数据集位姿对齐。

此时 target 在第 0 组相机坐标系，source 在第 5 组相机坐标系。目标是估计 `T_target_source`，使 source 进入 target 相机坐标系。

### 2. 调用自己的 ICP

将下采样点云转为 `(N, 3)` 数组。两片点云允许不同点数，行号不表示对应关系。

第一版初值单位矩阵，最大 50 轮、最大对应距离 0.10 米、RMSE 变化容差 1e-6。它们是本次起始设置，不是已证明最优的参数。

```python
T_own, history = icp_point_to_point(
    np.asarray(source_pcd.points),
    np.asarray(target_pcd.points),
    max_iterations=50,
    max_distance=0.10,
    tolerance=1e-6,
)
```

初始误差和重叠可能导致失败；失败也是应记录的结果。先记录原始问题，再决定如何修改设置。

### 3. 在相同点云上运行 Open3D 点到点 ICP

库基线的输入点云必须与自己的 ICP 一致，不能一边下采样、一边使用全量点。两边都从单位矩阵开始，使用同一距离上限和最大迭代数，点到点方法不估计缩放。

接口示例：

```python
reference = o3d.pipelines.registration.registration_icp(
    source_pcd,
    target_pcd,
    0.10,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1.0,
        relative_rmse=1e-6,
        max_iteration=50,
    ),
)
```

此处放宽 fitness 的变化门槛，让 RMSE 变化主要决定提前停止。当前 Open3D legacy 实现使用相邻轮差值比较。当前自实现记录轮初误差，但先更新 T 再判断停止，可能比库的更新后判断多执行一轮；此外 while 与库内部严格比较的边界也有差异。实验记录应说明这些规则；若比较同等停止条件，需要先统一停止时机。最终结果均用同一个外部评价器计算，不要求两种实现逐元素完全相同。

参考：[registration_icp API](https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_icp.html)、[v0.19 实现](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/registration/Registration.cpp)。新接口没见过时先看参数含义，再接入当前程序。

### 4. 显示、保存和共同几何评价

对比三种状态：单位矩阵初始状态、自己的 ICP 结果、Open3D 结果。两种算法都只变换 source，target 保持不动；显示时用副本，避免一次变换污染下一组输入。

使用固定的源/目标颜色、尽量相同的观察视角。保存代表性截图，以及两种方法配准后的点云；不新增一套重复配置文件。

用同一个外部评价过程评价初值和两种结果：变换原始 source，查找最近 target，按 0.10 米筛选，报告有效对应数量、有效比例和有效对应 RMSE。没有对应时报告失败，不把零 RMSE 当作成功。

这些几何指标没有使用真实对应标签，不足以单独证明位姿准确。当前先把真实帧对与同条件库对照跑通。

## 紧接着完成：参考相对位姿与误差

正式报告旋转、平移误差前，先把参考位姿配到每帧 RGB 时间：平移采用线性插值，旋转采用 SLERP；超出参考时间范围则跳过，不外推。RGB-depth 残余时间差也要记录，插值不消除传感器噪声和深度时间不同步。

### 当前任务：参考位姿评价器

在 `geometry.py` 中由学习者新增：

```python
def interpolate_pose(timestamp, pose_timestamps, poses):
    """输入查询时刻、(N,) 参考时间和 (N, 7) 位姿，返回该时刻的 (4, 4) T_wc。"""
```

参考位姿的每一行仍为 `[tx, ty, tz, qx, qy, qz, qw]`。查询时刻从 RGB 记录取得，不直接使用 `record['pose_index']` 对应的最近样本代替。

```python
source_time = rgb_records[source_record['rgb_index']][0]
target_time = rgb_records[target_record['rgb_index']][0]
```

实现步骤：

1. 检查查询时刻在参考时间首尾之间，否则明确报错；参考时刻已由读取函数排序，本任务正常输入应严格递增且至少两个样本。
2. 分别插值 tx、ty、tz。例如 x 使用 `np.interp(timestamp, pose_timestamps, poses[:, 0])`。这表示沿着已知时间—x 位置折线，读取查询时间的位置；y、z 同理。NumPy 默认的范围外端点填充不符合本任务不外推的约定，因此保留第 1 步检查。
3. 用 `Rotation.from_quat(poses[:, 3:7])` 得到各参考时刻的旋转，传入 `Slerp(pose_timestamps, rotations)`；调用 `slerp([timestamp]).as_matrix()[0]` 取得查询时刻的 3×3 旋转。不要直接对四元数四个分量逐项线性插值。
4. 将三个插值平移量和旋转填入单位 4×4 矩阵，返回 T_wc。

新接口只负责按时间取得参考位姿；不改动 ICP 的点对应、初值或更新过程。

小规模数学检查与这次函数一起验收：参考时间 0 秒、2 秒；位置分别 `[0,0,0]`、`[2,0,0]`，旋转分别为绕 Z 轴 0°、90°。查询 1 秒，应得到位置 `[1,0,0]` 与绕 Z 轴 45°的旋转；查询两个端点应恢复端点位姿；超出范围应报错。此处为预期检查值，不是已经完成的学习者实测。

学习入口：[NumPy interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html) 的 x/xp/fp 参数；下方 SciPy Slerp 的已知时刻、已知旋转、查询时刻示例。先理解线性插值直觉和接口，不要求自己推导 SLERP。

新知识学习入口：[SciPy Slerp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)。它在两个已知旋转之间插出指定时间的旋转，可直接调用库。到这一部分先解释接口与时间关系，再由学习者实现，不要求推导四元数插值公式。

若 `T_wc_source`、`T_wc_target` 分别表示两帧相机到数据集世界的参考变换，则：

```text
T_target_source_ref = inv(T_wc_target) @ T_wc_source
```

因为 source 点先到世界，再从世界到 target 相机坐标系。参考变换只用于画诊断对照与误差评价，不参与独立 ICP 的初始化和对应选择。

两次调用 interpolate_pose 得到 T_wc_source、T_wc_target 后，按上述公式计算 T_ref。这里得到的是与 ICP 相同方向的参考相对变换，而不是两帧位置直接相减。

在 `registration.py` 中由学习者新增：

```python
def evaluate_pose(T_est, T_ref):
    """返回旋转角误差（度）和平移向量误差范数（米）。"""
```

旋转部分：先计算 `R_delta = R_est @ R_ref.T`，通过 `Rotation.from_matrix(R_delta).magnitude()` 取得弧度，再用 `np.degrees()` 转成角度。平移部分计算 `norm(t_est - t_ref)`。这衡量相同 source→target 定义下的两种变换差异，不是三维点最近邻 RMSE。

学习入口：[Rotation.magnitude](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.magnitude.html)。自检同一变换与自己比较，误差应接近零；单位参考与“旋转 10°、平移 `[0.1,0,0]` 米”的已知变换比较，应得到约 10° 和 0.1 米。

最终对单位矩阵、自己的 T_own、Open3D 的 reference.transformation 分别调用 evaluate_pose，使用同一个 T_ref；与已有 fitness 和最近邻 RMSE 一起打印。完成后提供两个函数、接入代码、检查输出和真实帧对的结果。之后再扩展 source=10、20 及短序列。

届时报告两种方法相对于这个参考的旋转角误差与平移误差，并保持同一 source→target 方向；旋转矩阵元素最大差不是角度误差。

## 后续迁移和工程交付

- 用同一程序比较 `(source=5, target=0)`、`(source=10, target=0)`、`(source=20, target=0)` 三个帧对。保持变换方向为后续 source→第 0 组 target，记录 source/target 关联索引。
- 已增加命令行参数，支持数据目录、source/target 关联索引、体素大小、对应距离阈值、最大轮数、停止容差、输出目录和可选显示；自动保存三状态指标、变换矩阵、迭代记录和配准点云。
- 再扩展到短序列：每个新帧向上一帧配准并累计到首帧坐标，展示估计拼接与提供位姿拼接的差异，分析漂移。具体累计和轨迹评价协议在两帧通过后讲解。
- 最终整理真实帧对对照、失败案例与短序列，再由学习者执行第二关的 Git 提交和上传；助手检查内容与结果。

本任务与后续图文模型无关，先把几何配准做扎实。
