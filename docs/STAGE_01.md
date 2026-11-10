# 第一关：提供位姿下的 RGB-D 多帧融合

## 这一关的终点

用一条命令读取 TUM RGB-D 序列的 20—50 个采样帧，输出彩色融合点云、相机轨迹以及单帧/直接叠加/变换后融合的对照。

这次的新增能力是多帧时间关联和坐标统一。之前已经独立验证过的反投影代码可以迁移过来，不必重写一个单帧教学项目。

## 数据和环境

使用 `rgbd_dataset_freiburg1_desk` 的完整序列，不能只使用 Open3D 的单对样例。

- [官方序列下载页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)：搜索 `freiburg1_desk`，选 `tgz`，页面标注约 0.32 GB。
- [直接下载入口](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz)：可能跳转到官方 webshare 域名；已核对页面链接，未在本机下载验证。
- [官方文件格式](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)。

解压成：

```text
data/tum/rgbd_dataset_freiburg1_desk/
├── rgb/
├── depth/
├── rgb.txt
├── depth.txt
└── groundtruth.txt
```

第一关用现有 `yolov11` 环境；先执行 README 中的导入检查。此阶段是几何处理，可以先在本地 CPU 上做小规模验证。

## 今天先写的任务：三份索引的读取器

在项目根目录新建 `tum_io.py`。今天先实现下面两个函数和用于检查的 `__main__`，暂时不写融合大循环。

```python
def read_image_index(path):
    """返回按时间排序的 [(timestamp, relative_path), ...]。"""

def read_pose_index(path):
    """返回 timestamps: (N,)，poses: (N, 7)。
    poses 每行为 [tx, ty, tz, qx, qy, qz, qw]。
    """
```

处理空行及以 `#` 开始的注释行。用 `line.split()` 分割；时间戳和位姿转成数值，图片路径保留字符串。相对路径是相对于序列根目录，不是当前 Python 文件。

主程序依次读取三份文本，打印各自总数和前两条记录，再读取第一张 RGB 和第一张深度，打印各自 `shape`、`dtype`，核对路径可用。两张图此时仅用于读写检查，不能假设它们是已经按时间匹配的一对。

读取原始深度要保留 16 位，例如 `cv.imread(path, cv.IMREAD_UNCHANGED)`。本任务先使用 `pathlib.Path`、文件读取、`split()` 和列表；不需要 Pandas。

完成后由助手检查读取代码和实际输出，再继续时间匹配及变换，不把每个打印语句分成一关。

## 后续在本关写的核心内容

### 1. 时间关联

RGB、深度和 GT 轨迹有不同时间戳。不能按文件排序后的相同行号强行配对。

先自己实现带时间差上限的最近时间关联：RGB 与深度尽量一对一，避免重复复用同一张深度；关联成功后按时间排序。RGB-depth 最大差先设 0.02 秒，再为该 RGB 相机观测时间找最近 GT，最大差也先设 0.02 秒；超限则跳过。打印保留数、丢弃数和实际时间差。

这两个阈值是本项目的起始设置，不是数据集保证。最近 GT 只是第一版近似；正式比较估计位姿前将改成平移线性插值、旋转 SLERP，并记录协议。GT 可以供多帧使用，不要求一对一。不要将 RGB 与深度的轻微时间差当成完全同步。

### 2. 点云与坐标变换

在 `geometry.py` 放反投影和变换函数。TUM PNG 深度：`depth_m = depth_raw / 5000.0`，0 无效。第一版有效范围可选 `0 < Z < 4.0` 米，并记录选择。

沿用官方建议的 ROS 默认、不额外去畸变配置：`fx=fy=525.0, cx=319.5, cy=239.5`。这是本序列入门基线的明确选择，不是所有相机通用参数；TUM 原始深度已经做过比例校正，不再乘额外校正系数。

定义 `T_world_camera` 为把相机点变到世界点的 4×4 矩阵，记为 `T_wc`：

```text
p_world = R_wc @ p_camera + t_wc
```

TUM 的 `groundtruth.txt` 给出相机相对于世界的平移和四元数，按 `[qx,qy,qz,qw]` 传给 `Rotation.from_quat(...).as_matrix()`。四元数转换可用库，不要求你推导实现。点云若按 `(N,3)` 行向量存储，批量公式要自行根据维度改写。

相机坐标沿用 X 右、Y 下、Z 前；世界坐标由数据集定义，不要为了显示方便修改保存的数据。显示朝向和实际坐标变换是两件事。

### 3. 多帧融合入口

写 `fuse_known_poses.py`，调用前两个文件。先两帧确认，再扩到 20—50 个采样帧。用 Open3D 的 `voxel_down_sample` 控制点数，体素边长先取 0.02 米；区分直接拼接与体素下采样后的点数。

目标 CLI 接口，待你实现后使用：

```powershell
python fuse_known_poses.py --sequence data/tum/rgbd_dataset_freiburg1_desk --stride 10 --max-frames 30 --voxel-size 0.02 --output outputs/stage01
```

参数从关联后的有效帧列表采样。保存 PLY 和代表性截图；轨迹只需先画出相机中心连线及少量坐标轴，无须做复杂界面。

## 新知识怎么学

按用到的顺序学习，不用先看完整本 SLAM 教材：

1. **4×4 齐次变换、逆和连续变换**：看 [Open3D Transformation](https://www.open3d.org/docs/release/tutorial/geometry/transformation.html) 的 Translation、Rotation、General transformation。Open3D 可以做对照，主点变换用 NumPy 自己写。
2. **四元数接口**：看 [SciPy Rotation.from_quat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html)，只需先理解四个数表示旋转、输入顺序及转成旋转矩阵。
3. **时间戳与深度文件**：只读 TUM 文件格式中的 RGB/depth、Ground-truth trajectories 和 Intrinsic Calibration。
4. **体素下采样**：到点数开始增长时学习；先理解“固定三维小格内聚合点”的用途，再调用接口。

如果矩阵或时间匹配没见过，先把卡住的具体内容发来，助手先解释数据形状和一个小例子，再由你写回项目。

## 本关整体验收

- 数据读取、关联和变换核心代码由你完成，能说明每个数组的形状、单位和坐标系。
- 同一组帧给出单帧、直接叠加、变换后融合的截图；融合的大致结构一致，局部残差如实记录，不要求 GT 融合绝对无重影。
- 对点做 `T_wc` 变换再用其逆变换恢复，浮点误差量级合理（小坐标、float64 计算可用 1e-6 米作检查量级）；再与 Open3D 相同变换交叉检查。逆变换检查不能单独证明方向正确。
- 相机中心等于对应 `T_wc` 平移，能解释相对第一帧变换 `inv(T_wc_first) @ T_wc_i` 的含义。
- 独立迁移：改采样间隔和帧区间，无须手工修改路径或重新写主流程，能解释地图点数和细节的变化。
- 保留入口、关键参数、实际帧数、时间差统计及简短观察。助手协助整理为 GitHub 阶段记录。
