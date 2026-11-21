# 第四关：多视角特征与文本检索

整体目标是给地图实例关联多个 RGB 视角，提取并聚合图像特征，再用文本查找对应的 `map_id`，返回可核对的视角图。A 的视角导出、B 的图像特征与裁剪对照、C 的多视角聚合均已验收，当前进入 D 的文本查询。

## A 从三维实例导出二维观测图

在 `scripts/export_instance_views.py` 实现。沿用现有几何环境的 JSON 和 OpenCV，从地图记录回查已有检测缓存。

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
python -m scripts.export_instance_views --instances outputs/stage03_cli_chair/instances.json --detections outputs/stage03_chair_views_masks/results.json --crop-mode box --output outputs/stage04_chair_views

python -m scripts.export_instance_views --instances outputs/stage03_cli_chair/instances.json --detections outputs/stage03_chair_views_masks/results.json --crop-mode mask --output outputs/stage04_chair_views_mask
```

### 4. 裁剪实测

2026-11-20 验收。两组均导出 `map_id=1` 的三张 RGB 裁剪图，`views.json` 中的视角来源一致，`crop_mode` 分别为 `box` 和 `mask`。重新运行两条命令的像素与原有结果一致，回读 PNG 并与原图对应区域逐像素核对通过。

| 帧时间 | 检测框裁剪（宽×高） | 掩码外接框裁剪（宽×高） |
|---|---:|---:|
| `57390.089` | 157×175 | 154×175 |
| `57390.889` | 139×162 | 138×162 |
| `57391.689` | 118×172 | 112×170 |

两种方法都裁剪原色图像。第三帧的掩码外接框相对检测框从左侧收紧 6 像素，从顶部收紧 2 像素。这个样例测量了裁剪范围的变化，图文检索表现留到特征阶段比较。

## B 提取并缓存逐视角图像特征

在 `scripts/encode_instance_views.py` 实现。先用 `outputs/stage04_chair_views/views.json` 和它引用的 PNG。按实例及其 `views` 的顺序遍历，用 `views_path.parent / crop_file` 找到每张图。每张图编码成一个向量，保留向量行号与原视角的对应关系。

### 1. 先用单张图跑通模型接口

使用 `grounded_sam` 环境，已检查 Transformers 4.49.0、PyTorch 2.5.1 和 CUDA 可用。`google/siglip2-base-patch16-256` 在这个版本使用 `SiglipModel` 的固定 256 像素配置。脚本读取 `outputs/stage04_models/siglip2-base-patch16-256/`，目录内需要 `config.json`、`preprocessor_config.json` 和约 1.5 GB 的 `model.safetensors`。当前机器的官方权重已通过 SHA256 校验。[官方模型文件](https://huggingface.co/google/siglip2-base-patch16-256/tree/main) · [Transformers 4.49 图像特征接口](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/siglip#transformers.SiglipModel.get_image_features)

在可以直接访问 Hugging Face 的环境，从项目根目录下载脚本需要的三个文件：

```powershell
hf download google/siglip2-base-patch16-256 config.json preprocessor_config.json model.safetensors --local-dir outputs/stage04_models/siglip2-base-patch16-256
```

用 Pillow `Image.open(crop_path).convert("RGB")` 读取裁剪图。图像预处理器负责调整输入尺寸和像素归一化，模型输出的特征再单独做 L2 归一化。

```python
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

model_dir = Path("outputs/stage04_models/siglip2-base-patch16-256")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False, local_files_only=True)
model = AutoModel.from_pretrained(model_dir, use_safetensors=True, local_files_only=True).to(device).eval()

# images 是由一张或多张 PIL RGB 图像组成的列表。
inputs = processor(images=images, return_tensors="pt").to(device)
with torch.inference_mode():
    features = model.get_image_features(**inputs)
    features = torch.nn.functional.normalize(features.float(), dim=-1)
```

先打印单张图输出形状与范数，再把三张图一起传入。官方权重的单张 GPU 推理已实测得到 `(1, 768)`、范数 `1.0`，批量结果见下方实测。

### 2. 保存特征矩阵与行号索引

沿用 `argparse` 接收 `--views` 和 `--output`。用 `np.save` 将 `features.cpu().numpy()` 保存为 `view_embeddings.npy`，另存 `features.json`，顶层记录 `scene_id`、`model_id`、`crop_mode` 和 `rows`。每条 `rows` 记录包含 `feature_index`、`map_id`、`frame_id`、`local_id` 和 `crop_file`。从 0 开始编号，令 `rows[i]` 说明矩阵第 `i` 行是哪张图。

### 3. 验收与独立实现

实现后在项目根目录运行同一编码器的两组输入。

```powershell
conda activate grounded_sam

python -m scripts.encode_instance_views --views outputs/stage04_chair_views/views.json --output outputs/stage04_features_box

python -m scripts.encode_instance_views --views outputs/stage04_chair_views_mask/views.json --output outputs/stage04_features_mask
```

主任务检查检测框组重新读取后为 `float32` 的 `(3, 768)` 矩阵，所有数值有限、每行范数约为 1，与 JSON 三条来源一一对应。独立验收换到掩码外接框组，按 `(map_id, frame_id, local_id)` 配对，用归一化向量的点积计算每帧的余弦相似度。此处观察裁剪变化对特征的影响，下一步再聚合特征并加入多个实例做文本排序。

### 4. 逐视角特征实测

2026-11-21 验收。`box` 和 `mask` 两组均得到 `(3, 768)` 的 `float32` 矩阵，所有特征有限、每行 L2 范数约为 1。`features.json` 的 `feature_index` 与 `views.json` 的实例、帧和裁剪文件一一对应。用 `python -m experiments.compare_view_features` 复跑，结果如下。

| 帧时间 | 同帧 `box` 与 `mask` 余弦相似度 |
|---|---:|
| `57390.089` | 0.995959 |
| `57390.889` | 0.997202 |
| `57391.689` | 0.973099 |

第三帧是本组变化最大的一帧。掩码外接框从左侧收紧 6 像素、从顶部收紧 2 像素，用户观察到扶手边缘少了一点。分数衡量同帧两种裁剪的特征差异，检索准确度留到多实例文本查询时评估。

## C 聚合同一实例的多视角特征

新建 `scripts/aggregate_instance_features.py`，接收 `--features`（逐视角 `features.json` 路径）与 `--output`。从输入 JSON 同目录读取 `view_embeddings.npy`，按 `map_id` 收集各视角的 `feature_index`，然后按 ID 排序输出。对单个实例，按这些行号取向量，沿 `axis=0` 求均值，再做 L2 归一化。

```python
view_matrix = matrix[indices]
vector = view_matrix.mean(axis=0)
vector = vector / np.linalg.norm(vector)
```

`matrix[indices]` 一次取出若干行。每组的 `(视角数, 768)` 矩阵沿第 0 维求均值，得到一个 `(768,)` 向量。保存前确认范数大于 0，再用 `np.stack` 将每组向量叠成 `(实例数, 768)`，写入 `instance_embeddings.npy`。`instance_features.json` 保留 `scene_id`、`model_id`、`crop_mode`，以及实例行的 `feature_index`、`map_id`、`view_count`、`view_feature_indices`。

主样例用 `outputs/stage04_features_box/features.json`，输出到 `outputs/stage04_instances_box/`。实现后，从项目目录运行：

```powershell
python -m scripts.aggregate_instance_features --features outputs/stage04_features_box/features.json --output outputs/stage04_instances_box
```

验收为一个 `map_id=1`，三个视角，矩阵形状 `(1, 768)`，范数约为 1。现有三行的平均向量归一化前范数约为 `0.981475`。

独立验收复制这份输入到新目录，保留 `.npy` 的行顺序。把 JSON 里前两行的 `map_id` 改为 10，最后一行改为 42，再将 JSON 的 `rows` 重排为原顺序的 `[2, 0, 1]`。输出应有两个实例，视角数为 2 和 1，ID 42 的输出与其唯一输入向量一致（允许浮点误差）。此合成测试验证分组和索引，真实多物体图文检索将在加入场景中的其他物体后验收。

2026-11-21 实测。主样例输出 `(1, 768)`，ID `1` 来源为 `[0, 1, 2]`。独立输入的 JSON 行序为 `(42, 2)、(10, 0)、(10, 1)`，输出按 ID 形成 `(2, 768)` 矩阵，两行范数约为 1。ID `10` 的两视角向量与独立复算一致，ID `42` 与原始第 `2` 行的最大元素误差约为 `5.96e-8`。命令行重跑的输出与保存结果一致。

## D 用文本查找地图实例

新建 `scripts/query_instance_map.py`。输入一句英文物体描述，输出从高到低的 `rank`、`map_id`、余弦分数及来源 `view_feature_indices`。先用椅子地图跑通完整链路，再用同一场景中的椅子与床验证实际排序。

### 1. 补齐本地模型的分词文件

继续用既有 `grounded_sam` 环境和 `outputs/stage04_models/siglip2-base-patch16-256/` 中的权重。已在 Transformers 4.49.0 环境验证 `GemmaTokenizerFast` 与三份本地分词文件。新环境还未准备这些文件时，从项目根目录下载：

```powershell
hf download google/siglip2-base-patch16-256 tokenizer.json tokenizer_config.json special_tokens_map.json --local-dir outputs/stage04_models/siglip2-base-patch16-256
```

若 Hub CLI 的传输仍不稳定，沿用已验证过的官方 URL 和本地代理，通过 `curl.exe --fail --location --retry 2 --output <目标文件>` 下载对应文件，文件 URL 为 `https://huggingface.co/google/siglip2-base-patch16-256/resolve/main/<文件名>`。[官方文件列表](https://huggingface.co/google/siglip2-base-patch16-256/tree/main)

### 2. 编码文本并按分数排序

用 `argparse` 接收 `--features`（聚合后的 `instance_features.json`）、`--query`、`--top-k`。像前一步一样读取同目录的 `instance_embeddings.npy`。模型仍用本地 `AutoModel.from_pretrained(..., use_safetensors=True, local_files_only=True)`。查询转小写，按 SigLIP 2 训练设置固定补齐或截断到 64 个 token。[SigLIP 2 文本预处理](https://huggingface.co/docs/transformers/model_doc/siglip2#text-embeddings-and-retrieval) · [Transformers 4.49 文本特征 API](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/siglip#transformers.SiglipModel.get_text_features)

```python
from transformers import GemmaTokenizerFast

tokenizer = GemmaTokenizerFast.from_pretrained(model_dir, local_files_only=True)
tokens = tokenizer([query.lower()], padding="max_length", truncation=True,
                   max_length=64, return_tensors="pt").to(device)
with torch.inference_mode():
    encoded = model.get_text_features(input_ids=tokens["input_ids"])
    query_vector = torch.nn.functional.normalize(encoded.float(), dim=-1)[0].cpu().numpy()

scores = instance_embeddings @ query_vector
order = np.argsort(-scores)[:top_k]
```

`(实例数, 768) @ (768,)` 产生每个实例一个分数。按 `feature_index` 查输出排序行对应的 `map_id` 和来源。`top_k` 要大于 0，结果数最多为候选数。先验收文本向量 `(768,)`、范数约为 1，单椅结果返回 `map_id=1` 和三条来源索引。

```powershell
python -m scripts.query_instance_map --features outputs/stage04_instances_box/instance_features.json --query "a photo of a blue office chair" --top-k 1
```

已用正式权重和学习者的查询脚本实测：`rank=1 map_id=1 cosine=0.158133 views=[0, 1, 2]`。该输入只有一个候选，用于检查查询链路与来源回查。

### 3. 独立验收：同一场景的两个真实物体

已有椅子和床缓存时可直接运行下方的合并命令。若从零准备，在场景 `47333462` 的 `lowres_wide/` 中选出 `47333462_57390.089.png`、`47333462_57390.889.png` 和 `47333462_57391.689.png`，复制到 `outputs/stage03_chair_views_rgb/`，保留文件名。床使用 `57372.880` 帧。以下命令假设已有配置好 Grounding DINO＋SAM 的 `grounded_sam` 环境，且 `grounded_sam_project` 与本仓库同级；否则替换 `predict.py` 路径。

```powershell
conda activate grounded_sam
python ../grounded_sam_project/predict.py --source outputs/stage03_chair_views_rgb --text "a chair." --output outputs/stage03_chair_views_masks
python ../grounded_sam_project/predict.py --source data/arkitscenes/raw/Training/47333462/lowres_wide/47333462_57372.880.png --text "a bed." --output outputs/stage03_bed_masks
```

检查检测与叠加图，再切到 `yolov11` 环境运行 `python -m scripts.extract_instances`。当前提取脚本仍为单场景实验入口：换机器时将 `scene_dir` 指向本地解压的 `47333462`，按下表切换 `mask_dir` 和 `output_dir`。每次关闭点云窗口后检查 `observations.json` 与引用的 PLY。

| 提取组 | `mask_dir` | `output_dir` | 观测数 |
| --- | --- | --- | ---: |
| 椅子 | `outputs/stage03_chair_views_masks` | `outputs/stage03_observations` | 3 |
| 床 | `outputs/stage03_bed_masks` | `outputs/stage03_bed_observations` | 1 |

检测缓存会引用生成缓存时的图片路径，换机器时在本机重新生成。床的掩码覆盖不完整，点云仅保留可见部分；检索用原 RGB 的检测框裁剪。几何导出细节见[第三关 B](STAGE_03.md#b-的完整小项目逐帧三维实例观测导出器)。

运行 `scripts/make_retrieval_inputs.py`，将以下输入合并到 `outputs/stage04_retrieval_inputs/`。两个输入的 `scene_id` 和坐标系应一致。脚本用 `extend` 展开两份报告里的记录，使输出列表仍然是一条记录对应一个字典。

- 读取 `outputs/stage03_observations/observations.json` 和 `outputs/stage03_bed_observations/observations.json`，拼接 `observations`，把每条引用的 PLY 用 `shutil.copy2` 复制到新目录，保存合并的 `observations.json`。预期三条椅子观测加一条床观测，`frame_id` 和 `local_id` 沿用真实来源。
- 读取 `outputs/stage03_chair_views_masks/results.json` 和 `outputs/stage03_bed_masks/results.json`，拼接两份 `images`，保存为 `detections.json`。例如用 `dict(chair_report, images=chair_report["images"] + bed_report["images"])` 生成新的报告。本次用 `box` 裁剪，视角导出读取原图和检测框。

从项目目录按顺序准备输入、建图、导出视角、编码并聚合。合并、建图和导出用 `yolov11`，编码与查询用 `grounded_sam`。聚合只需要 NumPy，在两个环境都可以运行。

```powershell
conda activate yolov11
python -m scripts.make_retrieval_inputs

python -m scripts.instance_map --observations outputs/stage04_retrieval_inputs/observations.json --min_voxel_iou 0.10 --output outputs/stage04_retrieval_map

python -m scripts.export_instance_views --instances outputs/stage04_retrieval_map/instances.json --detections outputs/stage04_retrieval_inputs/detections.json --crop-mode box --output outputs/stage04_retrieval_views

conda activate grounded_sam
python -m scripts.encode_instance_views --views outputs/stage04_retrieval_views/views.json --output outputs/stage04_retrieval_features

python -m scripts.aggregate_instance_features --features outputs/stage04_retrieval_features/features.json --output outputs/stage04_retrieval_instances

python -m scripts.query_instance_map --features outputs/stage04_retrieval_instances/instance_features.json --query "a photo of a blue office chair" --top-k 2

python -m scripts.query_instance_map --features outputs/stage04_retrieval_instances/instance_features.json --query "a photo of a bed" --top-k 2
```

验收时固定同一份候选库。先用地图的来源确认哪个 ID 是椅子、哪个 ID 是床，再比较两次查询的 Top-1、两个分数和来源索引。类别标签只用来核对命中，排序完全依据余弦相似度。

### 4. 双实例检索实测

学习者用修正后的合并脚本完成独立运行，项目目录中已生成两份实例点云、四张视角裁剪和两行实例特征。床为 `map_id=1`（1 个视角），椅子为 `map_id=2`（3 个视角）。查询椅子时 ID 2 排第一（`0.158133`），查询床时 ID 1 排第一（`0.134168`）。两次使用同一份实例库。[来源核对、完整分数与限制](../reports/stage04_text_retrieval.md)
