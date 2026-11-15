# 第二关 B：自己实现点到点 ICP

启动日期：2026-11-11。验收日期：2026-11-13。本任务合成点云实现已通过，见 [实测记录](../reports/stage02_icp_synthetic.md)。下一步为 [真实 RGB-D 帧对](STAGE_02_RGBD.md)。以下保留实现任务供回顾。

## 这次的完整任务

在不知道逐行对应关系的两组点之间，迭代寻找最近邻并估计旋转和平移。输出最终 source→target 变换和每轮误差，随后在任务 C 中用于真实 RGB-D 帧对。

学习者自己实现循环、距离筛选、累计变换、停止条件和误差记录；可以调用 SciPy KD-tree 与已经写好的刚体拟合函数。当前不直接调用 Open3D ICP 代替自己的实现。

## 文件与接口

在 `registration.py` 中新增：

```python
def icp_point_to_point(
    source,
    target,
    max_iterations=50,
    max_distance=0.2,
    tolerance=1e-6,
):
    """返回最终 (4, 4) 的 source→target 变换 T，以及按明确约定记录的每轮 RMSE 列表 history。"""
```

在新建的 `demo_icp.py` 中生成数据、调用与评价。第一版从单位矩阵开始，之后真实数据阶段再按需要支持显式初始变换。

**误差记录口径说明：** 下方最初示范更新后记录，但每轮开始记录更新前误差也有效。当前学习者选择后者。需要在日志与说明中标清楚：history[k] 评价哪个 T；使用返回的最终 T 时，另外评价其最终分数。采用更新前误差且先更新再判断停止，会使停止时机滞后一轮，不能直接与更新后停止的迭代数等同。

## 数据：小位姿变化，并打乱目标顺序

```python
rng = np.random.default_rng(42)
source = rng.uniform(-1.0, 1.0, size=(1000, 3))
R_true = Rotation.from_euler("xyz", [3, -2, 4], degrees=True).as_matrix()
t_true = np.array([0.03, -0.02, 0.01])

target_clean = source @ R_true.T + t_true
target = target_clean[rng.permutation(len(target_clean))]
```

`target_clean` 保留已知对应顺序，只供最后评价。传给 ICP 的 `target` 已打乱顺序，算法必须自己找对应点。这里先使用无噪声数据，便于隔离 ICP 循环是否正确。

ICP 是局部配准方法；小角度、小平移用于先验证实现。这组数据已用 Open3D 参考实现自检可从单位矩阵恢复变换；这是出题数据检查，不是学习者 ICP 的结果，也不保证任意初始误差都能收敛。

## 新工具：KD-tree

在固定 target 上构建一次搜索结构：

```python
from scipy.spatial import cKDTree

tree = cKDTree(target)
distances, indices = tree.query(current_source, k=1)
```

- `current_source` 是 source 经当前累计 T 变换后的 `(N, 3)` 点集。
- `distances[i]` 是当前源点 i 到最近目标点的欧氏距离，不是距离平方。
- `indices[i]` 是最近目标点在 target 中的索引。
- 因此 `target[indices]` 按源点顺序排列出暂时的对应目标点。
- 多个源点允许找到同一个目标点；基础最近邻 ICP 不要求一对一分配，不套用 RGB-depth 的已使用集合。

学习入口：[SciPy cKDTree.query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)。当前只需掌握查询点、`k=1` 和两个返回值。

## 一种循环组织方式：更新后记录

1. 初始化累计 T 为 `np.eye(4)`，并创建空 history。
2. 每轮用累计 T 变换原始 source，得到 current_source。
3. 查询最近邻，保留距离不大于 max_distance 的对应。有效匹配不足 3 对时，明确报错并说明无法继续，避免将空对应送入拟合器；更完整的退化分布检查可在后续补充。
4. 把筛选后的 current_source 和对应 target 送给已有 `estimate_rigid_transform()`，得到本轮增量 delta_T。
5. 累计变换更新为 `T = delta_T @ T`。因为原始点先经历 T，再经历 delta_T，所以新增变换从左侧相乘。
6. 用更新后的 T 重新变换原始 source、查询距离并筛选，再计算本轮更新后的最近邻 RMSE，写入 history。若更新后有效对应不足，也应报告失败。
7. 达到 max_iterations，或相邻两轮 RMSE 的绝对变化小于 tolerance 时停止。第一轮没有上轮误差，不直接触发相邻轮停止条件。

每轮可以打印轮数、更新后的有效匹配数量和 RMSE，不要求额外配置文件或日志系统。

**重要约定：** 这里增量拟合的输入是已经变换后的 current_source。若改成拟合原始 source，得到的就不再是相同含义的增量，不能照用累计公式。

最近邻 RMSE：先过滤距离，再用 `sqrt(mean(valid_distances ** 2))`。它使用当前有效对应，不要求单调下降，也不能单独证明得到了正确变换。

max_distance 的单位为米。0.2 米只是本次坐标范围和初始误差下的起始值，不能自动沿用于所有真实序列。

## 评价与独立验收

调用 ICP 时只传 `source, target`。最终将 source 用 T_est 变换后：

- 打印估计的 R、t、真实值及二者误差；延续之前的旋转矩阵元素差和平移范数即可。
- 打印迭代次数、每轮 RMSE、最终有效匹配数。
- 用未打乱的 `target_clean` 计算最终真实对应点 RMSE；不能将 aligned 与打乱顺序的 target 逐行相减。
- 这组无噪声、一般位置点、小初始误差数据，可先以最终真实对应点 RMSE 小于约 1e-6 米检查实现。
- 独立迁移：保持点的位置不变，再次打乱 target 的顺序，配准结果应基本一致。说明为什么直接按行调用刚体拟合会失效，而 ICP 可以处理这次乱序。

本次不读取参考位姿来给出正确答案，不把 R_true、t_true 或 target_clean 送入 ICP 决策。它们只用于造数据和评价。

完成后检查代码、实际数值和数据流解释，再进入真实 RGB-D 帧对与 Open3D 对照。下一次工程整理与 GitHub 更新仍由学习者执行。

概念参考：[Open3D Point-to-point ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)。重点理解“寻找对应→拟合变换→更新”的循环，再对照库接口。
