# 第四关：多视角特征与文本检索

整体目标是给地图实例关联多个 RGB 视角，提取并聚合图像特征，再用文本查找对应的 `map_id`，返回可核对的视角图。A 先完成视角导出，后续在多个候选实例上验证文本排序。

## A 从三维实例导出二维观测图

在 `export_instance_views.py` 实现。沿用现有几何环境的 JSON 和 OpenCV，从地图记录回查已有检测缓存。

- 地图输入是 `outputs/stage03_cli_chair/instances.json`。椅子实例 `map_id=1` 有三条 `observations`，每条含 `frame_id`、`local_id`。
- 检测输入是 `outputs/stage03_chair_views_masks/results.json`。从 `image_info["image"]` 取得原 RGB 路径和文件主名，从 `detection["object_id"]` 取帧内检测编号。已检查这三条观测都能找到原图、框和掩码。

### 1. 用帧名和帧内编号查找检测

遍历 `results.json` 的 `images` 和每张图里的 `detections`，用 `(frame_id, object_id)` 做字典的键。值保存对应原图路径和检测字典。地图来源可用 `(observation["frame_id"], observation["local_id"])` 查到那条检测。同一个 `local_id` 可以在多帧重复，两个字段一起确定一条观测。

```python
frame_id = Path(image_info["image"]).stem
key = (frame_id, detection["object_id"])
lookup[key] = (image_info["image"], detection)
```

先用两个同编号的视角做小例子，检查查到的是指定帧。字典键是由帧名和检测编号组成的元组。

```python
lookup = {("frame_1", 1): "first.png", ("frame_2", 1): "second.png"}
obs = {"frame_id": "frame_2", "local_id": 1}
print(lookup[(obs["frame_id"], obs["local_id"])])  # second.png
```

### 2. 裁剪原 RGB 并保存视角索引

用 OpenCV 读原图。`box_xyxy` 的顺序是 `[x1, y1, x2, y2]`，值是浮点数。左上角向下取整，右下角向上取整，再把 X 限制在 `[0, width]`、Y 限制在 `[0, height]`。现有前两帧检测框的 `y1` 略小于 0，这一步会将其修正为 0。切片写成 `crop = rgb[y1:y2, x1:x2]`，保存前检查裁剪非空。

输出到 `outputs/stage04_chair_views/`。图像以 `instance_<map_id>_<frame_id>_<local_id>.png` 命名。另存 `views.json`，顶层记录 `scene_id` 和 `instances`。每个实例保存 `map_id`、`label` 和 `views`，每条视角至少含 `frame_id`、`local_id`、`crop_file`。这份视角索引会作为后续特征提取的输入。

### 3. 验收与独立实现

- 主任务导出 `map_id=1` 的三张椅子裁剪图，核对帧名、图片内容和 `views.json` 引用。检查同为 `local_id=1` 的三个视角都被保留。
- 独立实现用掩码的外接矩形做第二种裁剪。灰度读取 `mask_file` 后，用 `ys, xs = np.nonzero(mask > 0)` 找到前景坐标。左上角取 `xs.min()`、`ys.min()`，右下角取 `xs.max() + 1`、`ys.max() + 1`。检查掩码非空，再用相同的来源索引裁剪原 RGB，输出到另一个目录。比较两组裁剪覆盖的椅子与背景，保留定性差异供后续特征对照。

沿用 `argparse` 读取 `--instances`、`--detections` 和 `--output` 三个路径，用 `--crop-mode` 在 `box` 和 `mask` 之间切换。实现后，从项目根目录用现有几何环境运行同一份输入的两组对照。

```powershell
python export_instance_views.py --instances outputs/stage03_cli_chair/instances.json --detections outputs/stage03_chair_views_masks/results.json --crop-mode box --output outputs/stage04_chair_views

python export_instance_views.py --instances outputs/stage03_cli_chair/instances.json --detections outputs/stage03_chair_views_masks/results.json --crop-mode mask --output outputs/stage04_chair_views_mask
```
