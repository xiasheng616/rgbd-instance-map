# 第三关：跨帧三维实例地图

2026-11-15 启动，2026-11-19 更新。A 的数据适配与参考位姿重建、B 的逐帧实例导出及床样例迁移均已验收。C 已完成三帧椅子融合、合成双椅稳定 ID、近距离错合并对照和命令行复现。下一步从地图实例追溯二维观测，准备检索图像。[A 实测记录](../reports/stage03_arkit_preview.md) · [B 实测记录](../reports/stage03_instance_inputs.md) · [C 对照实测](#6-体素重叠对照实测) · [第四关任务](STAGE_04.md)

## 整关目标

在一个室内场景中，将多视角的物体观测合并成具有稳定 ID 的三维实例地图：再次看到同一把椅子时更新原实例，同类的另一把椅子保留独立 ID。先使用提供位姿隔离关联模块，再分析估计位姿误差的影响。

- A：接入一个 ARKitScenes 场景，完成 10 帧提供位姿重建。
- B：复用 Grounding DINO + SAM 2.1，缓存掩码，将二维实例转换到世界坐标系下的物体点云。
- C：自己实现几何候选筛选、一对一关联、新建与更新实例、体素融合；评价重复建图与错误合并。

模型推理可继续在 `grounded_sam` 环境执行，几何处理沿用已有几何环境，通过 mask PNG 和 JSON 衔接。缓存只需生成一次，再复用进行几何与关联实验。

## 已准备的数据

官方示例场景 `47333462` 已下载、解压并检查，完整记录见 [数据检查](../reports/stage03_data_check.md)。目录为：

```text
E:\shixi\rgbd_instance_map\data\arkitscenes\raw\Training\47333462
├── lowres_wide/             # RGB PNG
├── lowres_depth/            # uint16 深度 PNG，毫米
├── lowres_wide_intrinsics/  # 每帧 .pincam 内参
├── confidence/              # 深度置信度 PNG
├── lowres_wide.traj         # 提供的相机轨迹
└── 47333462_3dod_annotation.json
```

ZIP 原件保留在同目录。数据与 ZIP 被现有 `.gitignore` 排除。标注只用于后续检查，不将标注的物体 UID 当作预测关联答案。

## A 的完整小项目

新建 `arkit_io.py` 与 `preview_arkit.py`，由学习者实现。复用已有反投影、点变换、点云叠加和保存方式，保留当前熟悉的脚本风格。

### 1. 在 arkit_io.py 实现两个读取函数

```python
def read_intrinsics(path):
    """返回 K: (3,3)，以及整数 width、height。即 K, width, height。"""

def read_trajectory(path):
    """返回 timestamps: (N,)，T_wc_all: (N,4,4)，按时间排序。"""
```

`.pincam` 每行依次是 `width height fx fy cx cy`，使用这些值组装 K；不能沿用 TUM 固定的 525 内参。

`.traj` 每行依次是 `timestamp rx ry rz tx ty tz`。前三个旋转分量是旋转向量，其方向为旋转轴、长度为旋转角，单位弧度；不是欧拉角或四元数。

新接口学习：[SciPy Rotation.from_rotvec](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html)。本任务可以直接调用：

```python
R_cw = Rotation.from_rotvec(values[1:4]).as_matrix()
```

按 [ARKitScenes 官方加载器](https://github.com/apple-aiml-research/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py) 的约定，将该 R 与 `values[4:7]` 组装为世界到相机的 `T_cw`，再取逆得到 `T_wc`。这与原来 TUM 文件的布局、旋转表示和直接组装方式不同。

### 2. 关联 RGB、深度、内参和轨迹

以深度图文件为入口，按时间排序。RGB、深度、置信度在各自目录下使用同名 PNG；本场景的内参文件为相同主名加 `.pincam`。

图片主名形如 `47333462_57352.888`，下划线后的部分就是帧时间。对每帧寻找最近的参考轨迹记录，仅保留时间差 `< 0.005` 秒且文件齐全的记录。这个门槛沿用官方加载器的近时刻匹配思路；本场景保留下来的实际最大时间差约为 0.5 ms。

图像约 60 FPS，轨迹约 10 FPS，因此第一版保留与轨迹近乎同一时刻的子集。先实现此子集关联，无需对全部图像插值。

可增加 `associate_frames(scene_dir, pose_timestamps, max_diff=0.005)`，返回记录列表。每条至少包含 `timestamp`、`rgb_path`、`depth_path`、`intrinsics_path`、`confidence_path`、`pose_index` 和 `pose_dt`。此处路径统一为 Path 或统一为字符串即可，避免混用时忘记转换。

### 3. 在 preview_arkit.py 重建 10 帧

从关联记录中选取 `records[::10][:10]`。逐帧读取对应 K、RGB、深度与置信度，反投影得到相机坐标点云。

深度为毫米，使用 `Z = depth / 1000.0` 得到米。第一版保留 `0 < Z < 5` 米，并筛除置信度为 0 的像素。RGB 颜色仍需从 OpenCV BGR 转成 RGB。

每帧使用 `T_wc_all[record['pose_index']]`，调用已有 `transform_points(points_camera, T_wc)`，直接进入数据集世界坐标系。将 10 帧叠加，最后做一次 0.02 米全局体素下采样，保存到 `outputs/stage03_arkit_preview/reference_map.ply` 并显示。

本任务是提供位姿下的重建，不调用 ICP。未来实例观测也使用同一个场景世界坐标系。

## A 的检查与独立验收

- 打印各类文件数量、轨迹数量、关联成功数量，以及前两条关联与时间差。
- 打印首帧 RGB/depth 的形状和 dtype、从文件读取的 K，以及首个 T_wc。
- 提交 10 帧重建点云和实际显示结果；观察同一静态表面是否基本重合。
- 核对两个读取函数、关联代码和重建主循环，能解释为何改为除以 1000、为何不继续使用 525、为何轨迹要先组装 T_cw 再取逆。
- 独立迁移：将同一适配器用于关联记录的另一段 10 帧，保持从文件读取参数与位姿，提供结果，不为新帧手工重写 K 或变换矩阵。

A 的实现与独立区间检查已通过。既有脚本不整体重构；如需修改公共接口，先说明具体必要改动。

## B 的完整小项目：逐帧三维实例观测导出器

新建 `extract_instances.py`，由学习者实现。复用 arkit_io 的纯函数和已有 NumPy 反投影逻辑，将缓存分割掩码对应的有效深度像素转换为世界坐标点云，输出逐帧物体记录。稳定地图 ID 在 C 中通过关联产生。

### 已准备的输入

已复用既有 `grounded_sam_project/predict.py`，用本地缓存的 Grounding DINO Tiny 与 SAM 2.1 Hiera Tiny 完成三张椅子图像的推理。当前推荐输入为主体更完整的 `records[372]`、`records[380]`、`records[388]`，文件名保留原始数据名称。前两帧座面、扶手和脚轮较完整，第三帧补充椅背视角；仍有边缘裁切，使用可见部分进行多视角互补。

- RGB 缓存：`outputs/stage03_chair_views_rgb/`。
- 掩码、叠加图与检测 JSON：`outputs/stage03_chair_views_masks/`。
- JSON 入口：`outputs/stage03_chair_views_masks/results.json`。
- 几何脚本继续使用现有 `yolov11` 环境；本阶段无需再次加载模型。

当前输入的三张图各有一个椅子候选，掩码与深度尺寸均为 `(192,256)`。准备检查见 [缓存记录](../reports/stage03_instance_inputs.md)。这些候选是模型输出，不是准确率或完整物体真值。

### 自己写的处理过程

1. 读取场景轨迹与关联记录，以及缓存 JSON。
2. 外层遍历 `report['images']`，内层遍历每张图的 `detections`。JSON 的 `image` 指向 RGB 缓存副本，取文件名与原始场景记录对应；不要用完整路径直接比较。
3. 每帧读取原始 RGB、深度、置信度、内参，取得 `T_wc_all[record['pose_index']]`。
4. 使用 `mask_dir / detection['mask_file']` 读取灰度掩码，并用 `> 0` 转为布尔数组。先确认掩码、深度和置信度尺寸相同。
5. 在二维图上合并掩码与有效深度条件，再提取 XYZ 和对应 RGB：

```python
valid = mask & (Z > 0) & (Z < 5.0) & (confidence != 0)
```

6. 将筛选后的 `(N,3)` NumPy 点数组用已有 `transform_points(points_camera, T_wc)` 转到世界坐标，组装带 RGB 的点云，进行 0.02 米体素下采样。若没有有效点，明确跳过并输出原因。
7. 保存逐目标 PLY，并汇总每个观测的可见点云中心和世界坐标轴对齐包围盒。

注意：二维 mask 要与二维深度一同筛选。已有 `create_pcd()` 返回的点云经过无效像素过滤，不再有完整的 `(H,W)` 像素布局，不能直接用原始二维 mask 对它索引。这里需要沿用并调整你写过的反投影部分。

可以用文件名建立场景记录查找表：

```python
records_by_name = {record['rgb_path'].name: record for record in records}
record = records_by_name[Path(image_info['image']).name]
```

轴对齐包围盒 AABB 是分别取点云 X、Y、Z 的最小值和最大值形成的长方体。对下采样后准备保存的世界坐标点数组计算：

```python
saved_points = np.asarray(pcd.points)
centroid = saved_points.mean(axis=0)
bbox_min = saved_points.min(axis=0)
bbox_max = saved_points.max(axis=0)
```

此处 centroid 是当前可见点云的平均位置，AABB 也是当前观测的范围，均不视为完整物体的真实中心或完整尺寸。保存 JSON 时将 NumPy 向量转成普通列表。

### 输出约定

椅子组输出目录为 `outputs/stage03_observations/`。PLY 文件名由原始图片主名与从 0 开始的帧内遍历索引组成，例如 `47333462_57390.089_0.ply`，保留原始 RGB。后续通过 `cloud_file` 读取对应点云。

保存 `observations.json`，顶层包含 `scene_id`、`coordinate_frame='world'` 和 `observations` 列表。每条观测至少包含：

- `frame_id`：原始图片主名。
- `local_id`：缓存检测中的 object_id。
- `label`、`detection_score`。
- `point_count`：实际保存点云的点数。
- `centroid`、`bbox_min`、`bbox_max`：世界坐标，单位米。
- `cloud_file`：相对于输出目录的 PLY 文件名。

三张图的 local_id 当前都为 1；它仅在单张图片内有效。C 阶段再根据跨帧观测计算稳定的 map_id，不能直接沿用局部编号当跨帧 ID。

### B 的验收与独立迁移

- 读取缓存并导出三张图的椅子观测；按当前图像时间顺序，下采样前有效深度像素数应分别为 12898、10470、11964。保存后的点数按实际体素结果记录。
- 检查 PLY 使用世界坐标和 RGB，JSON 点数、中心与范围能从保存点云复算。
- 将三份点云同时显示，观察它们与各自原图掩码是否对应；保留可见部分不同、遮挡或漏分等现象。
- 独立迁移：换用 `outputs/stage03_bed_masks/results.json` 和同目录掩码，输出到另一个目录；相同几何代码应支持新类别和多张图。床的缓存已有覆盖部位不一致的失败现象，需要明确记录，不能因变成三维点云就认定语义正确。
- 验收时解释像素如何经过 mask、深度、K、T_wc 变成一个三维观测，以及为何 local_id 不能直接作为 map_id。

已有 `preview_arkit.py` 在导入时会执行预览主流程。新脚本先使用 `arkit_io.py` 和 `geometry.py` 的函数，反投影部分沿用现有写法；不直接导入预览入口来触发显示和保存。

三帧椅子实际导出了 1671、1789、1923 个下采样后的点。床样例迁移也生成了三份 PLY，坐标、RGB 与观测统计均通过复核。掩码漏掉的床面在几何导出时无法恢复，详见 [B 实测记录](../reports/stage03_instance_inputs.md)。

## C 跨帧关联与实例融合

**小项目：把三帧看见的同一把椅子合成一个地图实例，同时给不同的同类物体保留各自的 ID。** 在 `instance_map.py` 中实现按帧关联和点云融合。主样例复用 B 的世界坐标观测，三帧依次为 `57390.089`、`57390.889`、`57391.689`。

### 1. 读取观测并按帧分组

- 输入 `outputs/stage03_observations/observations.json`。每条记录用 `frame_id` 标识帧，用 `local_id` 标识该帧内的检测。
- 用字典按 `frame_id` 分组，可以用 `frames.setdefault(frame_id, []).append(obs)`。取出帧名最后一段作为时间戳，例如 `float(frame_id.rsplit("_", 1)[1])` 得到 `57390.089`，再按其数值从小到大遍历帧。
- 用 `json_path.parent / obs["cloud_file"]` 得到该条记录的 PLY 路径，交给 `o3d.io.read_point_cloud`。点已在同一个世界坐标系，直接用于距离计算和融合。

### 2. 对每一帧分配或复用地图 ID

用字典 `instances` 保存地图实例，以整数 `map_id` 为键。实例里记录类别 `label`、Open3D 点云 `cloud`、当前融合点云质心 `centroid`，以及来源列表 `observations`。来源记录保留 `frame_id` 和 `local_id`。新 ID 从 1 开始递增。

1. 进入一帧时，用 `available_ids = set(instances)` 记录帧开始前已存在的地图 ID。这个集合在进入下一帧时重新建立。
2. 对该帧的每个观测，从 `available_ids` 中找同类别的候选。将观测的 `centroid` 转成 NumPy 数组，调用 `np.linalg.norm(obs_center - map_center)` 计算两个质心之间的距离，单位是米。
3. 选择距离最小且不超过 `0.5 m` 的候选。匹配后从 `available_ids` 移除该 ID，使它本帧只接收这一条观测。第一帧以及找不到合格候选的观测都新建 ID，新建 ID 从下一帧开始参与匹配。
4. 每条观测打印帧名、`local_id`、分到的 `map_id`，以及新建或匹配。匹配时附上实际使用的距离，便于定位错误。

`0.5 m` 是这个小项目的起步门槛。用现有 PLY 按下面的融合方式复算，第二帧到第一帧质心约 `0.019 m`，第三帧到前两帧融合质心约 `0.312 m`。同一物体的 `local_id` 可以跨帧变化，关联依据是类别和空间位置。

### 3. 合并点云并更新实例

新建实例直接接收当前观测的 PLY。匹配到已有实例时，将它的点云与当前观测叠加，再做一次 `0.02 m` 体素下采样。已有点云与当前观测均保留 RGB。

```python
merged_cloud = (old_cloud + observation_cloud).voxel_down_sample(0.02)
```

从融合后的点计算新质心，写回该实例，下一帧使用更新后的质心比较距离。新建或匹配都要把当前观测的 `frame_id`、`local_id` 加到来源列表里。

### 4. 输出实例地图并检查主样例

输出到 `outputs/stage03_instance_map/`。每个实例保存 `instance_<map_id>.ply`；`instances.json` 的顶层包含 `scene_id`、`coordinate_frame`（`world`）、`match_distance_m`（`0.5`）、`voxel_size_m`（`0.02`）和 `instances` 列表。列表里的每条记录包含 `map_id`、`label`、`cloud_file`、`point_count`、`centroid`、`bbox_min`、`bbox_max` 和来源 `observations`。从要写出的融合点云计算统计值，向量用 `.tolist()` 转成 JSON 数组。

主样例应得到一个 `instance_1.ply`，来源列表依时间记录三帧。运行后检查新建和匹配日志、显示带 RGB 的融合结果，再读取保存的 PLY 复算点数、质心和包围盒，与 JSON 比对。

### 5. 独立验收：两帧两把同类椅子，交换检测编号

用主样例的前两帧构造独立输入，原观测记为 A。每帧各复制一份椅子 PLY，整体沿世界坐标 X 轴平移 `+0.40 m` 作为 B，保持类别 `a chair`，同步更新复制件的 `cloud_file`、质心和包围盒。把合成输入和输出放在各自的目录。将第二帧的两条记录放到第一帧记录之前写入 JSON，顺便检查时间排序。

| 帧 | `local_id=1` | `local_id=2` |
|---|---|---|
| `57390.089` | 原椅子 A | 平移椅子 B |
| `57390.889` | 平移椅子 B | 原椅子 A |

验收时应保留 **2 个地图实例，每个有 2 条来源**。A 的来源是第一帧 `1` 和第二帧 `2`，B 的来源是第一帧 `2` 和第二帧 `1`。本例实测正确对应相距约 `0.019 m`，跨实例候选约 `0.40 m`，两类距离都通过 `0.5 m` 门槛。结果用于检查同帧一对一和跨帧 ID，真实同类家具的关联另用场景观测验证。

主样例实测得到一个实例、三条来源和 2621 个融合点。双椅测例实测得到两个实例，每个实例有两条来源和 2197 个融合点。两条来源链分别为 `1 → 2` 和 `2 → 1`，地图 ID 稳定，PLY 点数、质心和包围盒与输出 JSON 一致。

### 6. 体素重叠对照实测

使用同一份输入，仅保留帧 `57390.089` 的原椅子 A 和帧 `57390.889` 的平移椅子 B。程序在第二帧只有 A 一个候选。两者的质心距离为 `0.405769 m`，在共享的 `0.05 m` 世界坐标体素网格上，占据体素的 IoU 为 `0.037084`。两次运行均使用 `0.5 m` 距离门槛和 `0.02 m` 的融合体素。

| 匹配条件 | 地图实例数 | 每个实例的来源数 | 每个实例的点数 |
|---|---:|---|---|
| 只用质心距离 | 1 | 2 | 2808 |
| 质心距离加 IoU ≥ 0.10 | 2 | 1、1 | 1671、1789 |

已核对两份 `instances.json` 引用的 PLY、点数、质心、包围盒和来源。当前输入位于 `outputs/stage03_observations_test/`，产物分别位于 `outputs/stage03_instance_map_test_only_distance/` 和 `outputs/stage03_instance_map_test_Iou_distance/`。该对照使用一把椅子的真实点云与平移副本，验证了这个受控案例中的错合并。床样例仍用于分析分割覆盖变化带来的质心漂移，真实多家具场景的关联另行验证。

### 7. 复现实验的命令行入口

`instance_map.py` 接收观测 JSON 路径、输出目录和可选 `--min_voxel_iou`。脚本按 JSON 所在目录解析 `cloud_file`，省略门槛时使用质心距离，指定 `0.10` 时再应用体素重叠门槛。输出 `instances.json` 记录实际设置，分别为 `null` 和 `0.1`。

```powershell
python instance_map.py --observations outputs/stage03_observations_test/observations.json --output outputs/stage03_cli_distance

python instance_map.py --observations outputs/stage03_observations_test/observations.json --min_voxel_iou 0.10 --output outputs/stage03_cli_iou

python instance_map.py --observations outputs/stage03_observations/observations.json --min_voxel_iou 0.10 --output outputs/stage03_cli_chair
```

前两组命令复现了同一输入下的一个和两个实例。补齐参数记录后，在临时目录复跑检查了两个门槛值。第三组是独立输入迁移，实际保存了一个实例、三条来源、2621 个带 RGB 的融合点。回读 PLY 复算的质心和包围盒与 JSON 一致。

## 数据与接口来源

- [ARKitScenes 数据格式与下载](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md)
- [RAW 数据组织](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/README.md)
- [官方位姿转换与点云加载示例](https://github.com/apple-aiml-research/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py)
