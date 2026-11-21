# 第二关：从对应点刚体拟合到 ICP 配准

启动日期：2026-11-10。更新日期：2026-11-15。状态：A 对应点拟合、B 合成 ICP、C 真实帧对与短序列的基础实现和实验检查已完成，包含采样对照及换起点独立迁移。当前 ICP 基线存在明显漂移，失败区间保留。见 [A 实测记录](../reports/stage02_rigid_fit.md)、[B 实测记录](../reports/stage02_icp_synthetic.md)、[C 实现说明](STAGE_02_RGBD.md) 和 [短序列结果](../reports/stage02_sequence.md)。Git 发布由学习者执行。

## 本关最终交付

一个可诊断的两帧配准器，以及使用估计相对变换的短序列拼接。学习者自己实现刚体拟合、最近邻对应筛选、点到点 ICP 迭代和误差计算；NumPy SVD、KD-tree、Open3D 显示与基线可以使用。

| 顺序 | 完整任务 | 验收重点 |
|---|---|---|
| A：已通过 | 已知一一对应点的刚体拟合器 | 从点对恢复已知旋转和平移；去中心化、旋转约束、噪声残差 |
| B：已通过 | 未知对应点的点到点 ICP | 交替找对应与估计增量，正确累计变换，有停止条件和失败信息 |
| C：基础实现及实验已检查 | 真实 RGB-D 帧对与短序列 | 与 Open3D 在相同输入、初值和阈值下对照，报告旋转/平移误差及失败案例 |

先把一个完整的对应点拟合任务做通，再接 ICP；不为每个语法或打印项单独设关。

## A：你这次写什么

新增两个文件：

- `rgbd_map/registration.py`：放可复用的 `estimate_rigid_transform(source, target)`，由你实现核心算法。
- `experiments/demo_rigid_fit.py`：生成已知答案的点对、调用估计器、评估并可选显示。

先使用当前环境在 CPU 运行，不需要新的数据集或模型依赖。复用 `rgbd_map/geometry.py` 中的 `transform_points()`。

接口约定：

```python
def estimate_rigid_transform(source, target):
    """输入两个 (N, 3) 数组，第 i 行是一对已知对应点。
    返回 (4, 4) 的 T_target_source，将 source 变换到 target。
    """
```

几何公式采用列向量：`q = R @ p + t`。你的点集按行存储，因此批量公式仍为 `target ≈ source @ R.T + t`。这次返回的 T 表示 source→target，不自动表示相机→世界。

**对应关系是本任务的前提。** `source[i]` 对应 `target[i]`。不能把两张 RGB-D 图生成的点云按行号直接配对：它们的点数、可见表面和像素对应通常不同；那是后面 ICP 要处理的问题。

## 生成具有已知答案的数据

把下面的数据生成部分放进 `experiments/demo_rigid_fit.py`，核心估计函数仍由你写：

```python
import numpy as np
from scipy.spatial.transform import Rotation

rng = np.random.default_rng(42)
source = rng.uniform(-1.0, 1.0, size=(100, 3))

R_true = Rotation.from_euler("z", 30, degrees=True).as_matrix()
t_true = np.array([0.3, -0.2, 0.1])
target = source @ R_true.T + t_true
```

含义：100 个三维点，绕 Z 轴旋转 30 度，再平移 `[0.3, -0.2, 0.1]` 米。目标点的顺序保持不变，因此对应关系已知。

`R_true`、`t_true` 只用于生成数据和检查答案。估计函数只能接收两组点，不能读取真值变量。

## 核心算法：SVD / Kabsch 刚体拟合

刚体拟合寻找统一的旋转和平移，最小化所有对应点之间的平方距离；不加入缩放或剪切。

### 1. 求两组点的质心

质心就是各坐标的平均值。对 `(N, 3)` 点集使用 `mean(axis=0)`，得到 `(3,)` 的 `[平均 X, 平均 Y, 平均 Z]`。

记 source 质心为 `c_source`，target 质心为 `c_target`。

### 2. 去中心化

每个点减去本组质心，得到 `P` 和 `Q`，形状仍为 `(N, 3)`。

因为 `c_target = R @ c_source + t`，所以 `q - c_target = R @ (p - c_source)`。平移消掉后，可以先估计旋转。

### 3. 计算相关矩阵并做 SVD

```text
H = P.T @ Q         # (3, N) @ (N, 3) → (3, 3)
H = U @ diag(S) @ Vt
```

可直接调用 `U, S, Vt = np.linalg.svd(H)`。SVD 是一个数值分解工具；先理解它把 H 分解为三个部分，以及怎样组合出旋转，暂不要求完整证明。

NumPy 文档中第三个返回值叫 `Vh`；这里 H 为实数，所以它就是 `V.T`，代码中记为 `Vt`。不要把它误认为 V 本身。

### 4. 恢复旋转，并排除镜像

对应上面的 H 定义，候选旋转为：

```text
R = V @ U.T
V = Vt.T
```

检查 `det(R)`。正常旋转的行列式应接近 `+1`；`-1` 表示反射，不能作为刚体运动的旋转。

如果候选行列式小于 0，将 `Vt` 的最后一行取反，再用修正后的 `Vt` 重新计算 R。它是在旋转约束下修正 SVD 解，不是把整个 R 取反。

### 5. 恢复平移

```text
t = c_target - R @ c_source
```

质心本身也经历旋转，不能只计算 `c_target - c_source`。

### 6. 组装 4×4 矩阵

复用第一关的组装方式：左上角填 R、右上角填 t、最后一行是 `[0, 0, 0, 1]`。返回 T。

## 检查与验收

调用自己写的估计器，并复用 `transform_points(source, T_est)` 得到 `aligned`。

至少打印：

- 真实与估计的 R、t。
- `R.T @ R` 和 `det(R)`：分别接近单位矩阵和 +1。
- R 的最大元素绝对差、平移向量误差的范数。
- 配准前、配准后的对应点 RMSE。

这里的对应点 RMSE 定义为：每对点三维距离的平方，先对点取平均，再开根号。

```python
rmse = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))
```

数组流：`(N, 3)` 坐标差 → `(N,)` 距离平方 → 标量平均值 → 开根号，单位仍是米。这与对全部坐标元素直接做均方根有一个 sqrt(3) 的比例差，报告时采用这里的定义。

本次无噪声、100 个一般位置三维点、float64 情况，目标是配准后 RMSE 与平移误差低于约 `1e-8` 米，旋转矩阵最大元素差低于约 `1e-8`。这是合成数学正确性检查，不是真实传感器精度。

**独立迁移验收：** 换成包含多个轴的旋转和另一组平移，再向 target 加入标准差 0.002 米的独立高斯噪声，用同一个估计函数报告恢复结果和残差。噪声下不要求误差为零，也不预设某个固定测量值。说明已知对应关系、去中心化和噪声残差各自的意义。

第一版在一般位置的合成点上验证；后续再处理退化分布、错误对应和离群点。阶段演示入口用 `if __name__ == "__main__":` 保护，可复用算法放进函数。材料由助手协助整理，提交和上传由学习者执行。

## 没见过这些知识时怎么学

1. **质心与去中心化**：先看本页步骤 1—2，手算两三个点的均值并验证去中心化后均值接近零，再用于当前拟合任务；不单设新的提交或验收。
2. **SVD 接口**：看 [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) 的二维矩阵说明、返回值和重构示例。只需先掌握 `U, S, Vt` 的含义和形状。
3. **Kabsch 的任务**：看 [SciPy align_vectors](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html) 开头的目标函数与 Kabsch 说明。这里拟合的是去中心化的点坐标，不把每个点各自单位化；本次核心函数也不直接调用这个完整求解器。
4. **ICP**：当前任务通过后，再看 [Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html) 的 Point-to-point ICP。它交替寻找对应点与调用刚体拟合；本次写的函数会成为其内部模块。

不需要先看完一门 SLAM 课。遇到不理解的矩阵，先问它表示什么、形状是什么，再写回代码。

## 后面 B、C 的边界

- ICP 从原始相机坐标点云出发。若先用参考位姿对齐再宣称 ICP 独立估计成功，会混淆实验条件。
- 参考位姿用于评价相对变换；真实数据正式旋转/平移误差比较前，完善时间插值与变换方向约定。
- 与 Open3D 对照固定点云、下采样、初始变换、距离阈值和停止条件，并记录运行失败。
- ICP 是局部方法，需要检查初始误差、重叠和对应质量。提供 GT 初值的诊断结果单独标注。
