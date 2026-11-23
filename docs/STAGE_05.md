# 第五关：批量检索评估与多视角对照

**小项目：用固定的查询与候选库评估三维实例检索。** 第四关手动检查了两句查询；第五关先把正确答案、排序、失败记录和指标写成可复跑的流程，再比较视觉输入的差异。

- A：复用现有特征库，批量计算 Top-1 和 MRR，确认评估本身没有错。
- B：保持候选 ID、查询、编码器和裁剪方式相同，只比较每个实例的最早有效视角与全部视角聚合。用帧时间确定首视角，不按查询得分挑图。
- C：保持协议不变，换未用于调整方案的场景，报告命中与失败案例。当前的双候选场景用于开发和调试，不作为跨场景能力结论。

## A 批量检索评估

### 1. 输入与输出

你写 `scripts/evaluate_retrieval.py`，不修改第四关的查询、编码、建图或聚合代码。准备好的查询表位于 [experiments/fixtures/stage05_queries.json](../experiments/fixtures/stage05_queries.json)：场景 `47333462` 中，床是 `map_id=1`、椅子是 `map_id=2`。前两句沿用第四关，其余是同两个物体的不同描述。在看本轮分数前固定标签和文本；跑完不要删掉失败的查询。

- `--features`：第四关的 `instance_features.json`；同目录的 `instance_embeddings.npy` 是候选库。
- `--queries`：含 `scene_id` 和 `queries` 的 JSON。每条含 `query_id`、`text`、`target_map_id`。
- `--output`：输出目录。写 `per_query.json`（每条保留查询字段、`top1_map_id`、`top1_cosine`、`target_rank`）及 `summary.json`（`scene_id`、`num_queries`、`num_candidates`、`top1_accuracy`、`mrr`）。

这里是**闭集**评估：每条查询的正确对象都应在候选库中。两份 JSON 的场景 ID 要一致，`feature_index` 要恰好覆盖矩阵行号、每个 ID 对应一个候选，查询集不能为空。如果目标 ID 不在库里，报出对应的 `query_id`，不要跳过或当作有效负例。

### 2. 你要实现的流程

1. **批量编码。** 参考 [第四关的查询脚本](../scripts/query_instance_map.py) 的本地加载与预处理。不要直接 import 该脚本，它在模块顶层解析命令行。新入口本次只加载一次模型，给 tokenizer 传文本列表；仍转小写、补齐或截断到 64 token，输出沿最后一维做 L2 归一化。不加 `[0]`，保留每条查询的向量。
2. **矩阵打分。** 查询矩阵形状是 `(Q, 768)`，实例矩阵形状是 `(N, 768)`。一行对应一句查询，一列对应候选矩阵的一行。

   ```python
   scores = query_matrix @ instance_embeddings.T  # (Q, N)
   ```

3. **求真实目标的完整排名。** 写一个不加载模型的 `evaluate_scores(scores, feature_rows, queries)`。先用 `feature_index` 建索引，把列号转换成 `map_id`。对每行可用 `np.argsort(-score_row, kind="stable")` 降序排序；同分时保留较小的特征索引在前。转换成 ID 列表后，在**完整排名**中找 `target_map_id`，从 1 开始计名次。真实标签只用来算指标，不参与排序。
4. **统计与落盘。** Top-1 为排名第一的查询比例；MRR 为每条查询的正确名次倒数的平均值。先逐条算 `1 / target_rank`，再取平均，不是用平均名次取倒数。把 NumPy 标量用 `int(...)` / `float(...)` 转成 Python 标量再写 JSON。把模型加载和命令行解析放进函数，从 `if __name__ == "__main__":` 调用入口，便于导入纯评估函数做测试。

### 3. 运行真实数据

在项目根目录运行。复用已准备的模型权重、分词器和候选特征，不重新下载或提取图像特征。

```powershell
conda activate grounded_sam
python -m scripts.evaluate_retrieval --features outputs/stage04_retrieval_instances/instance_features.json --queries experiments/fixtures/stage05_queries.json --output outputs/stage05_retrieval_a
```

验收：分数矩阵为 `(6, 2)`，逐条报告有 6 条。前两条的 Top-1 应复现 ID 2 和 ID 1；批量与逐句编码的相似度可有小数误差。其余四条和汇总以实跑为准。六句话对应两个物体，即使全部命中，也不能当成六个独立实例或泛化准确率。

### 4. 独立验收：把正确答案故意放到第二或第三

把下列例子喂给你写的纯评估函数（不加载模型）。候选矩阵列顺序是 ID 10、42、90，但 JSON 元数据行顺序故意打乱。

```python
import numpy as np

feature_rows = [
    {"feature_index": 2, "map_id": 90},
    {"feature_index": 0, "map_id": 10},
    {"feature_index": 1, "map_id": 42},
]
scores = np.array([[0.90, 0.50, 0.10],
                   [0.60, 0.80, 0.20],
                   [0.80, 0.90, 0.70]])
queries = [
    {"query_id": f"q{i}", "text": f"synthetic query {i}", "target_map_id": target}
    for i, target in enumerate([10, 10, 90], start=1)
]
```

正确名次应为 `[1, 2, 3]`，Top-1 为 `1/3`，MRR 为 `(1 + 1/2 + 1/3) / 3 = 11/18 ≈ 0.611111`。这个用例同时检查了列号映射和失败项没有从统计中漏掉。交付你写的入口、真实两份输出和这组独立验收结果，再进入 B。

### 5. A 阶段实测

学习者的正式输出保存在 `outputs/stage05_retrieval_a/`：6 条查询针对 2 个候选，逐条核对正确目标名次均为 1，Top-1 和 MRR 均为 1.0。独立入口 `python -m experiments.check_retrieval_metrics` 通过了三候选乱序样例，返回名次 `[1, 2, 3]`，并独立核算 Top-1 = 1/3、MRR = 11/18。合成案例和正式入口已分开。

## B 首视角与多视角对照

### 1. 实现首视角输入构造

由学习者新建 `experiments/make_first_view_features.py`。参数 `--features` 指向第四关的逐视角 `features.json`，需要其同目录的 `view_embeddings.npy`；`--output` 为新的首视角输入目录。先写纯函数 `select_first_views(rows)`，只负责挑选行，不写文件：

1. 按 `map_id` 把记录分组。复用 `by_map.setdefault(row["map_id"], []).append(row)`。
2. 对每组用 `min(group, key=...)` 获取时间最早的记录。帧时间的取法：

   ```python
   time_s = float(row["frame_id"].rsplit("_", 1)[1])
   ```

   如果时间完全相同，用较小的 `feature_index` 稳定地打破并列。不要按 JSON 行的顺序、图片质量或查询得分选择。
3. 按 `map_id` 排序所选记录，将原 JSON 的 `rows` 替换为这些行，保留模型、场景、裁剪方式与其他元数据。原 `view_embeddings.npy` 直接用 `shutil.copy2` 复制到新目录，不重排、不缩减，也不重编号 `feature_index`。后续聚合器只读取新 JSON 里引用的行。入口放在 `if __name__ == "__main__":` 下，便于单独测试选帧函数。

### 2. 运行对照

沿用 A 的查询、候选、模型、裁剪方式和评估。只将多视角实例特征库替换成首视角库。在项目根目录的 `grounded_sam` 环境运行：

```powershell
python -m experiments.make_first_view_features --features outputs/stage04_retrieval_features/features.json --output outputs/stage05_first_view_inputs
python -m scripts.aggregate_instance_features --features outputs/stage05_first_view_inputs/features.json --output outputs/stage05_first_view_instances
python -m scripts.evaluate_retrieval --features outputs/stage05_first_view_instances/instance_features.json --queries experiments/fixtures/stage05_queries.json --output outputs/stage05_retrieval_first
```

预期首视角特征来源为床 ID 1 的 `[0]`，椅子 ID 2 的 `[1]`；多视角原库是 `[0]` 和 `[1, 2, 3]`。两边候选数都必须为 2。用 `query_id` 对齐两份 `per_query.json`，比较每条的 `target_rank` 和汇总的 Top-1 / MRR。若排名都没变，结论就是这六条查询没有观察到排名收益，不把分数波动写成准确率提升。床本来只有一个视角，实际改变的只有椅子的表示。

### 3. 独立验收

直接把同一份 `rows` 的副本用 `list(reversed(rows))` 倒序后，传给你写的 `select_first_views`。不需要再建一个测试文件，临时调用两次即可。两次都应返回相同的 `{map_id: feature_index}`，即 `{1: 0, 2: 1}`。这个场景里如果错误地拿每组第一条，倒序后椅子会选成 `3`，因此它确实能检验是否按时间选帧。

要区分「最早时间」和「最小特征索引」，再用这两条记录调用同一个函数。应选索引 `9`，因为 `1.0` 秒比 `2.0` 秒早：

```python
test_rows = [
    {"map_id": 7, "feature_index": 1, "frame_id": "demo_2.0"},
    {"map_id": 7, "feature_index": 9, "frame_id": "demo_1.0"},
]
assert select_first_views(test_rows)[0]["feature_index"] == 9
```

### 4. B 阶段实测

核对了两份真实落盘报告、首视角清单和矩阵。首视角用了特征行 `[0]` 和 `[1]`，多视角用了 `[0]` 和 `[1, 2, 3]`。首视角输入的矩阵与原逐视角矩阵逐字节一致，两个实例库均为 `(2, 768)`。床的实例向量相同；椅子在两个条件下的实例向量余弦约为 0.989277。倒序输入和时间与索引反序的小例子均验收通过。

| 条件 | 床 / 椅子视角数 | Top-1 | MRR | 正确目标排名第一 |
| --- | --- | ---: | ---: | ---: |
| 首视角 | 1 / 1 | 1.0 | 1.0 | 6/6 |
| 全部视角聚合 | 1 / 3 | 1.0 | 1.0 | 6/6 |

六条查询的目标名次都未变化。按查询对齐后，三条椅子查询都预测同一个椅子 ID，其首视角余弦分别为 0.163644、0.142903、0.125971；多视角分别为 0.158133、0.135957、0.120204。这里得分变化没有带来排名变化。这个开发场景只有两个候选，且只有一把椅子真正改变了视角数，不能据此断言首视角或多视角普遍更好。

## C 迁移到未调参的场景

先将既有 `scripts/extract_instances.py` 做成可复用入口。**本步不新建 Python 文件、不改几何或分割规则。** 由学习者用 `argparse` 接入：

- `--scene`（`type=Path`）：场景目录，赋给现有 `scene_dir`，由 `scene_dir.name` 保持场景 ID。
- `--masks`（`type=Path`）：包含 `results.json` 和掩码图片的目录，赋给 `mask_dir`。
- `--output`（`type=Path`）：输出目录，赋给 `output_dir`。
- `--show`（`action="store_true"`）：只有显式开启才调用 `draw_geometries`，默认只落盘。

参数接入后先用熟悉的椅子缓存回归，不覆盖已有结果。在项目根目录运行：

```powershell
conda activate yolov11
python -m scripts.extract_instances --scene data/arkitscenes/raw/Training/47333462 --masks outputs/stage03_chair_views_masks --output outputs/stage05_c_extract_regression
```

这条命令只读取现有 RGB-D、位姿和椅子掩码，输出新的 `observations.json` 和三份 PLY，不重跑检测。与 `outputs/stage03_observations/` 对照：场景 ID、三条 `(frame_id, local_id)` 和每条点数（1671、1789、1923）应一致，且默认不弹点云窗口。

回归后再选另一访次的数据，以不为该场景修改算法阈值为前提比较两组检索。按 [ARKitScenes 官方数据说明](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md) 先确认标注，再按场景下载所需 RGB-D、置信度、内参与轨迹，不下载整库。届时还需要核对查询表与特征库的 `scene_id` 一致：`map_id` 只在本场景内有意义，不能沿用开发场景的答案编号。

### 场景参数回归实测

最终版正例 `outputs/stage05_c_extract_recheck/` 的 `observations.json` 与第三关逐项一致，三份 PLY 的 SHA-256 分别与原版相同。传入不存在的场景路径时，脚本在读取该路径下的 `lowres_wide.traj` 时报 `FileNotFoundError`，未生成报告。此项验证的是场景参数没有被旧目录覆盖，不是留出场景检索结果。

### 在下载帧之前筛选标注

[官方 Raw 划分](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/raw_train_val_splits.csv)中，开发序列 `47333462` 属于访次 `467138`。候选 `47333431`、`47333440` 和 `47333452` 均为 Validation，且各自不属于开发访次。以下地址来自[官方下载脚本](https://github.com/apple-aiml-research/ARKitScenes/blob/main/download_data.py)。这里只读取小的标注 JSON，按类别计数，不下载图像或写入本地文件。网络读取失败不等于场景缺少这个文件。

```powershell
$arkitValidation = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation'
foreach ($sceneId in '47333431','47333440','47333452') {
    $body = curl.exe -fsSL --connect-timeout 8 --max-time 20 --max-filesize 1048576 "$arkitValidation/$sceneId/${sceneId}_3dod_annotation.json"
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not read scene $sceneId"; continue }
    $annotation = ($body -join "`n") | ConvertFrom-Json
    $labels = $annotation.data | Group-Object label | ForEach-Object { "$($_.Name): $($_.Count)" }
    "$sceneId => $($labels -join ', ')"
}
```

先找有至少三个容易区分的物体、优先包含开发场景已用过的类别（如椅子）的候选。标注数不等于有效候选数；只下载一个合适场景，之后还要确认同一实例至少在两帧有有效观测。选择在看 SigLIP 查询得分之前确定，避免用分数挑测试场景。

### 留出场景 47333452：下载与预览

根据学习者读取的标注，固定选择 Validation 序列 `47333452`（访次 `467172`），其中有椅子 5 把、床 1 张、桌子 4 张。相比其他两个筛过的候选，它保留了开发时的椅子和床类别，并能增加真实候选。在查看检索结果前确定场景；无需把所有标注对象或所有 RGB 帧都送进模型。

在项目根目录的 PowerShell 运行。下载命令只取这个序列的四个 ZIP、轨迹和标注。下载失败时保留未完成的 `.part`，不会将其当作正式文件；重跑会跳过已经下载的正式文件。

```powershell
$sceneDir = 'data/arkitscenes/raw/Validation/47333452'
$sceneUrl = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47333452'
$modalities = @('lowres_wide', 'lowres_depth', 'confidence', 'lowres_wide_intrinsics')
$files = @($modalities | ForEach-Object { "$_.zip" }) + @('lowres_wide.traj', '47333452_3dod_annotation.json')
New-Item -ItemType Directory -Path $sceneDir -Force | Out-Null
foreach ($file in $files) {
    $dest = Join-Path $sceneDir $file
    if (Test-Path -LiteralPath $dest -PathType Leaf) { continue }
    curl.exe -fL --retry 2 --connect-timeout 15 --max-time 1800 -o "$dest.part" "$sceneUrl/$file"
    if ($LASTEXITCODE -ne 0) { throw "下载失败：$file" }
    Move-Item -LiteralPath "$dest.part" -Destination $dest
}
foreach ($modality in $modalities) {
    if (Test-Path -LiteralPath (Join-Path $sceneDir $modality) -PathType Container) {
        Write-Host "保留已有目录：$modality"
        continue
    }
    Expand-Archive -LiteralPath (Join-Path $sceneDir "$modality.zip") -DestinationPath $sceneDir -ErrorAction Stop
}
```

解压到场景根目录，不额外套一层模态子目录。若解压中断，不要仅凭目录存在认定文件齐全；预览前检查关联成功数和缺配套文件的提示。

下面的命令不保存 Python 源文件，只在 `yolov11` 环境复用现有的帧关联逻辑，从具备配套深度、置信度、内参和邻近位姿的 RGB 里等间隔抽最多 24 张，写出 `outputs/stage05_c_47333452_preview/contact_sheet.jpg`。编号指 `associate_frames` 返回列表的索引。

```powershell
conda activate yolov11
@'
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from rgbd_map.arkit_io import associate_frames, read_trajectory

scene = Path("data/arkitscenes/raw/Validation/47333452")
times, _ = read_trajectory(scene / "lowres_wide.traj")
records = associate_frames(scene, times)
if not records:
    raise SystemExit("No associated RGB-D frames. Check the extracted files.")

count = min(24, len(records))
indices = [round(i * (len(records) - 1) / max(count - 1, 1)) for i in range(count)]
sheet = Image.new("RGB", (6 * 220, ((count + 5) // 6) * 187), "#202631")
draw = ImageDraw.Draw(sheet)
for slot, index in enumerate(indices):
    x, y = slot % 6 * 220, slot // 6 * 187
    path = records[index]["rgb_path"]
    with Image.open(path) as rgb:
        sheet.paste(ImageOps.contain(rgb.convert("RGB"), (212, 159)), (x + 4, y + 4))
    draw.text((x + 4, y + 168), f"{index}: {path.stem.rsplit('_', 1)[1]}", fill="white")
output = Path("outputs/stage05_c_47333452_preview/contact_sheet.jpg")
output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output, quality=92)
print(f"associated_frames={len(records)}, preview={output}")
'@ | python -
```

看图后优先选 6–10 张配套齐全的帧，目标为至少三个可区分的真实物体、有一个物体在两帧以上可见。先固定帧和类别，再运行检测、点云提取与建图；检索查询答案用新场景实例 ID，不沿用旧场景编号。

### 留出场景 47333452：固定选帧、分割与建图

实际关联成功 `516` 帧。按 RGB 预览与配套数据、在运行模型前冻结以下 8 个关联索引：

| 区域 | 关联索引 | 时间戳（秒） | 选取理由 |
| --- | --- | --- | --- |
| 客厅 | 134、157、161 | 54965.617、54967.916、54968.316 | 保留客厅灰椅的重复视角；157 和 161 还可见邻近另一把灰椅 |
| 床边 | 291、313 | 54981.311、54983.510 | 保留同一张床的不同视角，并覆盖冰箱一侧 |
| 厨房 | 358、368、381 | 54988.008、54989.008、54990.324 | 多次看到厨房同一把白椅，部分帧包含冰箱 |

这只是可见性预检，尚不代表分割正确、成功建成独立 ID，或每个物体都成功关联。固定提示词为 `a chair. a bed. a refrigerator.`；保留开发阶段的检测阈值、深度反投影与地图 IoU 门槛。下列命令不修改源码：

```powershell
$rgbDir = 'data/arkitscenes/raw/Validation/47333452/lowres_wide'
$selectedRgb = 'outputs/stage05_c_47333452_rgb'
$stamps = @('54965.617','54967.916','54968.316','54981.311','54983.510','54988.008','54989.008','54990.324')
New-Item -ItemType Directory -Path $selectedRgb -Force | Out-Null
foreach ($stamp in $stamps) {
    $name = "47333452_${stamp}.png"
    Copy-Item -LiteralPath (Join-Path $rgbDir $name) -Destination (Join-Path $selectedRgb $name) -ErrorAction Stop
}

conda activate grounded_sam
python ../grounded_sam_project/predict.py --source outputs/stage05_c_47333452_rgb --text "a chair. a bed. a refrigerator." --box_threshold 0.35 --text_threshold 0.25 --output outputs/stage05_c_47333452_masks

conda activate yolov11
python -m scripts.extract_instances --scene data/arkitscenes/raw/Validation/47333452 --masks outputs/stage05_c_47333452_masks --output outputs/stage05_c_47333452_observations
python -m scripts.instance_map --observations outputs/stage05_c_47333452_observations/observations.json --min_voxel_iou 0.10 --output outputs/stage05_c_47333452_map
python -m scripts.export_instance_views --instances outputs/stage05_c_47333452_map/instances.json --detections outputs/stage05_c_47333452_masks/results.json --crop-mode box --output outputs/stage05_c_47333452_views
```

下载目录存放原始数据；第一段仅复制固定的 8 张 RGB，保留原文件名。检测输出掩码，提取脚本用同名深度/位姿转成世界坐标点云，地图按既有逻辑关联，最后的裁剪图用于核实每个 `map_id` 对应哪个物体。初验先看掩码叠加和地图来源：至少三个物理上可辨的候选、至少一个真实物体有两条以上正确关联的观测。若同一把椅子被拆成两个 ID、不同椅子被合并或有效对象不足，应记录失败，不根据检索得分换帧或调阈值。多把椅子都符合 `a chair` 时，不把这个含糊查询硬标成唯一 ID。

### 留出场景 47333452：候选审查与检索协议

8 张图产生 17 条二维检测。帧 `54989.008` 的整张置信度 PNG 为 0，其冰箱和椅子两条检测均没有通过有效点筛选。其余 15 条三维观测映射成 8 个地图 ID。按实际裁剪图、原图叠加和来源记录，人工核对如下：

| 地图 ID | 物理对象 | 观测数 | 审查结论 |
| --- | --- | --- | --- |
| 1、3 | 同一个浅木色抽屉柜 | 1 / 2 | 两组都误标，分别为 `a bed` 与 `a bed a refrigerator`；原样作为干扰项 |
| 2 | 客厅灰色打孔椅 | 2 | 两条观测指向同一把椅子 |
| 4、8 | 同一张床 | 2 / 1 | ID 8 只截到床边，记录为重复/拆分 |
| 5、6 | 同一台浅色冰箱 | 2 / 2 | 二维前端在同帧 `54981.311` 就有重叠检测（掩码 IoU 约 0.752），地图中保留了两个 ID |
| 7 | 厨房白色打孔椅 | 3 | 与灰椅不同，三条观测指向同一把椅子 |

保留全部 8 个原始地图候选，首视角和全视角不做人工删减，不修改检测或关联阈值。检索评价单位是**真实物体**，而不是某一个预选的地图编号。因此一个物体允许对应多个正确 ID；取其中最先出现的名次计算 Top-1 和 MRR。地图重复和检测误标另外记录，这两个指标不代表地图质量。

由学习者改已有的 [scripts/evaluate_retrieval.py](../scripts/evaluate_retrieval.py)，让查询可用 `target_map_ids` 列表。旧入口仍接受 `target_map_id`。每条查询检查正确 ID 集合非空、所有 ID 都存在于候选库，然后在完整排序中取第一个命中的名次。指标公式、图文编码和地图代码不改。例：正确集合是 `{4, 8}`，排名为 `[8, 7, 4]` 时名次为 1，排名为 `[7, 8, 4]` 时为 2。

由学习者新建 `experiments/fixtures/stage05_queries_47333452.json` 保存以下固定查询。正样本来自查看点云和裁剪图，不根据检索得分增删候选或修改答案。

```json
{
  "scene_id": "47333452",
  "queries": [
    {"query_id": "gray_chair_01", "text": "a gray perforated chair", "target_map_ids": [2]},
    {"query_id": "gray_chair_02", "text": "a chair with a gray seat and white legs", "target_map_ids": [2]},
    {"query_id": "white_chair_01", "text": "a white chair with a perforated seat", "target_map_ids": [7]},
    {"query_id": "white_chair_02", "text": "a white chair beside the stove", "target_map_ids": [7]},
    {"query_id": "bed_01", "text": "a bed with a dark blue blanket", "target_map_ids": [4, 8]},
    {"query_id": "bed_02", "text": "a bed with pink patterned bedding", "target_map_ids": [4, 8]},
    {"query_id": "fridge_01", "text": "a cream-colored refrigerator", "target_map_ids": [5, 6]},
    {"query_id": "fridge_02", "text": "a refrigerator with pink stickers", "target_map_ids": [5, 6]}
  ]
}
```

在项目根目录的 `grounded_sam` 环境运行。先复跑旧单答案测试，再编码 15 张已导出的视角。全视角直接聚合；首视角复用同一份编码、每个地图 ID 只选最早观测再聚合。末尾分别评估并保存报告。

```powershell
conda activate grounded_sam
python -m experiments.check_retrieval_metrics

python -m scripts.encode_instance_views --views outputs/stage05_c_47333452_views/views.json --output outputs/stage05_c_47333452_features
python -m scripts.aggregate_instance_features --features outputs/stage05_c_47333452_features/features.json --output outputs/stage05_c_47333452_multi_instances
python -m experiments.make_first_view_features --features outputs/stage05_c_47333452_features/features.json --output outputs/stage05_c_47333452_first_inputs
python -m scripts.aggregate_instance_features --features outputs/stage05_c_47333452_first_inputs/features.json --output outputs/stage05_c_47333452_first_instances

python -m scripts.evaluate_retrieval --features outputs/stage05_c_47333452_first_instances/instance_features.json --queries experiments/fixtures/stage05_queries_47333452.json --output outputs/stage05_c_47333452_eval_first
python -m scripts.evaluate_retrieval --features outputs/stage05_c_47333452_multi_instances/instance_features.json --queries experiments/fixtures/stage05_queries_47333452.json --output outputs/stage05_c_47333452_eval_multi
```

验收：两库均含相同 8 个候选，每个库是 `(8, 768)`，两份逐条报告各有 8 条查询。另用合成排名检查多答案逻辑，确认命中 ID 8 时不会被误判为没命中 ID 4。保留实际排名失败项，不预设哪个模式更高。

### C 阶段实测

学习者已完成两组评估。逐项核对到 15 条视角特征、相同的 8 个原始地图 ID、相同的 8 条查询与多答案标注；两个实例矩阵均为 `(8, 768)`，按源索引复算与落盘结果一致。两组 Top-1 都是 8/8，MRR 都是 1.0，没有查询改变第一个正确结果的名次。查询 `fridge_02` 的第一名 ID 从 5 换到 6，但两者都是事先核定的同一台冰箱。[逐条结果与报告](../reports/stage05_retrieval_comparison.md)

### 第二个留出场景：相似物体或部分遮挡

固定候选为 ARKitScenes Validation 序列 `47333440`，访次 `467176`，不同于开发场景的 `467138` 与已评估留出场景的 `467172`。[官方划分](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/raw_train_val_splits.csv#L3441-L3450)。学习者先前读到的标注包含三把椅子、五个柜子以及洗衣机、马桶和水槽；当前仅确认标注类别，不确认相似椅子的可见性、遮挡或可用配对帧。

先采集下列画面证据，再冻结该场景的选帧，不根据检索得分挑样本：

1. 找到两把可独立辨认、视觉相近的椅子，最好至少一帧同时出现；每把最好有两帧有效观测。目标是确认两个实体，不把同一把椅子的两个地图 ID 当成同类干扰。
2. 如果同一把椅子有较早的部分遮挡或画面截断，以及较晚的更清楚视角，记录两帧及其差别。首视角仍定义为时间最早的有效观测，不能在评分后挑最低分的视角。遮挡（被物体挡住）和截断（物体超出画面）分别记录。
3. 查询用裁剪图中可核对的特征来区分椅子，例如座面颜色、椅背有无孔洞或椅脚形状。若现有裁剪里两把椅子确实无法区分，不硬给含糊描述指定一个 ID。未检出、串 ID、重复 ID 等另记为上游问题；不把漏检目标当作闭集检索样本。

两组都沿用同一份原始地图、所有有效候选、相同文本与答案、相同模型和阈值。对比 Top-1、MRR、逐条排名变化，并额外数清楚「查 A 却把 B 排第一」的同类混淆。没有满足相似或遮挡条件时，可报告该场景不符合本次难例协议，不把它描述成已验证的遮挡测试。

#### 下载与预览

以下在项目根目录的 PowerShell 运行。前半段只下载指定序列的低分辨率 RGB、深度、置信度、内参、轨迹和标注；后半段在场景根目录解压。下载地址与文件命名来自[官方下载脚本](https://github.com/apple-aiml-research/ARKitScenes/blob/main/download_data.py)。不会改动之前的场景。

```powershell
Set-Location E:\shixi\rgbd_instance_map
$hardSceneId = '47333440'
$hardSceneDir = "data/arkitscenes/raw/Validation/$hardSceneId"
$hardSceneUrl = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/$hardSceneId"
$hardModalities = @('lowres_wide', 'lowres_depth', 'confidence', 'lowres_wide_intrinsics')
$hardAssets = @($hardModalities | ForEach-Object { "$_.zip" }) + @('lowres_wide.traj', "${hardSceneId}_3dod_annotation.json")
New-Item -ItemType Directory -Path $hardSceneDir -Force | Out-Null
foreach ($asset in $hardAssets) {
    $dest = Join-Path $hardSceneDir $asset
    if (Test-Path -LiteralPath $dest -PathType Leaf) { continue }
    curl.exe -fL --retry 2 --connect-timeout 15 --max-time 1800 -o "$dest.part" "$hardSceneUrl/$asset"
    if ($LASTEXITCODE -ne 0) { throw "下载失败：$asset" }
    Move-Item -LiteralPath "$dest.part" -Destination $dest
}
foreach ($modality in $hardModalities) {
    if (Test-Path -LiteralPath (Join-Path $hardSceneDir $modality) -PathType Container) { continue }
    Expand-Archive -LiteralPath (Join-Path $hardSceneDir "$modality.zip") -DestinationPath $hardSceneDir -ErrorAction Stop
}
```

下载后用下面的命令复用帧关联，等间隔抽最多 36 张已经配对的 RGB。它只生成预览图，不创建源码、不跑分割、不计算检索分数。图上编号是关联列表索引。

```powershell
conda activate yolov11
@'
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from rgbd_map.arkit_io import associate_frames, read_trajectory

scene = Path("data/arkitscenes/raw/Validation/47333440")
times, _ = read_trajectory(scene / "lowres_wide.traj")
records = associate_frames(scene, times)
if not records:
    raise SystemExit("No associated RGB-D frames. Check the extracted files.")

count = min(36, len(records))
indices = [round(i * (len(records) - 1) / max(count - 1, 1)) for i in range(count)]
sheet = Image.new("RGB", (6 * 220, ((count + 5) // 6) * 187), "#202631")
draw = ImageDraw.Draw(sheet)
for slot, index in enumerate(indices):
    x, y = slot % 6 * 220, slot // 6 * 187
    path = records[index]["rgb_path"]
    with Image.open(path) as rgb:
        sheet.paste(ImageOps.contain(rgb.convert("RGB"), (212, 159)), (x + 4, y + 4))
    draw.text((x + 4, y + 168), f"{index}: {path.stem.rsplit('_', 1)[1]}", fill="white")
output = Path("outputs/stage05_c_47333440_preview/contact_sheet.jpg")
output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output, quality=92)
print(f"associated_frames={len(records)}, preview={output}")
'@ | python -
```

预览验收：关联帧数大于零，预览图可打开且带索引。之后结合预览和邻近帧，定约 6–10 张用于难例对照；不提前假设候选数量或检索实验结果。

#### 47333440 预览裁决：不纳入独立难例评估

该序列成功关联 524 帧。对照原图 `47333452_54994.722.png` 和 `47333440_53842.927.png`：前者从厨房方向拍到红白瓷砖浴室和门上虎纹毯，后者从同一门口反向看见此前的绿色玩具、浅木抽屉柜、粉色收纳盒和地毯。这是两个访次很可能属于同一套住处的画面证据。不同访次编号并不足以核实物理环境独立。

该浴室只确认一把成人灰椅的连续视角，画面中的小脚凳和黄色儿童玩具不构成相似成人椅对；灰椅能找到远景和画面边缘截断，但不能把门旁的毯子说成挡住了椅子。保留原始下载和预览用于记录排除理由；本序列不计入独立难例结果，不运行主检索或为其调整阈值。

#### 替补：标注初筛后，先只下载 RGB 预览

首选 `42446540`（访次 `422381`），备用 `42897564`（访次 `422785`），都列在 [ARKitScenes 官方 Validation 划分](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/raw_train_val_splits.csv)。[ReVSI 的公开计数标注](https://huggingface.co/datasets/3dlg-hcvc/ReVSI)分别记录 4 把和 7 把椅子，比没有椅子类别线索的随机候选更适合预览。这不是对 Apple 标注数量、同帧可见性或实际遮挡的核验。可选地先读 Apple 标注，**不写文件**：

```powershell
$altValidation = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation'
foreach ($sceneId in '42446540','42897564') {
    $body = curl.exe -fsSL --connect-timeout 8 --max-time 20 --max-filesize 1048576 "$altValidation/$sceneId/${sceneId}_3dod_annotation.json"
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not read scene $sceneId"; continue }
    $annotation = ($body -join "`n") | ConvertFrom-Json
    $labels = $annotation.data | Group-Object label | ForEach-Object { "$($_.Name): $($_.Count)" }
    "$sceneId => $($labels -join ', ')"
}
```

按固定顺序先只看 `42446540`，不在此时下载备用场景。下面只生成一个场景的 `lowres_wide.zip`、解压后的 RGB 和一张预览图。生成的预览是 `outputs/stage05_c_42446540_rgb_preview/contact_sheet.jpg`。不新建 Python 源文件。

```powershell
Set-Location E:\shixi\rgbd_instance_map
$altSceneId = '42446540'
$altRoot = "data/arkitscenes/raw/Validation/$altSceneId"
$altRgbArchive = Join-Path $altRoot 'lowres_wide.zip'
New-Item -ItemType Directory -Path $altRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $altRgbArchive -PathType Leaf)) {
    $altRgbUrl = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/$altSceneId/lowres_wide.zip"
    curl.exe -fL --retry 2 --connect-timeout 15 --max-time 1800 -o "$altRgbArchive.part" $altRgbUrl
    if ($LASTEXITCODE -ne 0) { throw 'RGB 下载失败' }
    Move-Item -LiteralPath "$altRgbArchive.part" -Destination $altRgbArchive
}
if (-not (Test-Path -LiteralPath (Join-Path $altRoot 'lowres_wide') -PathType Container)) {
    Expand-Archive -LiteralPath $altRgbArchive -DestinationPath $altRoot -ErrorAction Stop
}

conda activate yolov11
@'
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

scene_id = sys.argv[1]
rgb_dir = Path("data/arkitscenes/raw/Validation") / scene_id / "lowres_wide"
frames = sorted(rgb_dir.glob("*.png"))
if not frames:
    raise SystemExit("No RGB files. Check the extracted directory.")
count = min(36, len(frames))
indices = [round(i * (len(frames) - 1) / max(count - 1, 1)) for i in range(count)]
sheet = Image.new("RGB", (6 * 220, ((count + 5) // 6) * 187), "#202631")
draw = ImageDraw.Draw(sheet)
for slot, index in enumerate(indices):
    x, y = slot % 6 * 220, slot // 6 * 187
    with Image.open(frames[index]) as rgb:
        sheet.paste(ImageOps.contain(rgb.convert("RGB"), (212, 159)), (x + 4, y + 4))
    draw.text((x + 4, y + 168), frames[index].stem, fill="white")
out = Path("outputs") / f"stage05_c_{scene_id}_rgb_preview" / "contact_sheet.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=92)
print(f"rgb_frames={len(frames)}, preview={out}")
'@ | python - $altSceneId
```

这是只看 RGB 的预览，标题标的是文件名，未做深度、置信度或位姿配对，不能把打印的 `rgb_frames` 作为可用关联帧数。确认有不同环境、可验证的相似物体或前景遮挡后，再补齐同版本的剩余资产并配对。

#### 42446540 预览通过：补齐 RGB-D 并冻结选帧

已核对本地 5367 张原始 RGB 和预览：灰色沙发、阳台和白色马赛克厨房与 `47333452` 的布置明显不同。餐桌两侧能看见多把花纹近似的餐椅；`201499.999` 是条纹夹克、深色外套覆盖椅背的清楚证据，`201506.996` 可见另一把花纹椅，部分被桌面/桌布挡住。衣物在扫视中没有被移走的证据，不称为后来脱离遮挡。

从已有 RGB 预先确定参考时刻：`201472.994`、`201473.294`、`201475.393`、`201499.999`、`201500.499`、`201501.998`、`201506.996`、`201508.996`。前三个保持共现和第一侧视角，中间三个保留挂衣椅子和相邻花纹椅，后两个提供桌边另一角度。此时只验证了 RGB 可见性，未验证配套数据与正确的地图关联。

在项目根目录运行。下面只补深度、置信度、内参、轨迹和标注，**跳过已经下载和解压的 RGB**。ZIP 解压到场景目录。文件命名依据 [ARKitScenes 官方下载脚本](https://github.com/apple-aiml-research/ARKitScenes/blob/main/download_data.py)。

```powershell
Set-Location E:\shixi\rgbd_instance_map
$sceneDir = 'data/arkitscenes/raw/Validation/42446540'
$sceneUrl = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446540'
$remainingModes = @('lowres_depth','confidence','lowres_wide_intrinsics')
$remainingAssets = @($remainingModes | ForEach-Object { "$_.zip" }) + @('lowres_wide.traj','42446540_3dod_annotation.json')
foreach ($asset in $remainingAssets) {
    $dest = Join-Path $sceneDir $asset
    if (Test-Path -LiteralPath $dest -PathType Leaf) { continue }
    curl.exe -fL --retry 2 --connect-timeout 15 --max-time 1800 -o "$dest.part" "$sceneUrl/$asset"
    if ($LASTEXITCODE -ne 0) { throw "下载失败：$asset" }
    Move-Item -LiteralPath "$dest.part" -Destination $dest
}
foreach ($mode in $remainingModes) {
    if (Test-Path -LiteralPath (Join-Path $sceneDir $mode) -PathType Container) { continue }
    Expand-Archive -LiteralPath (Join-Path $sceneDir "$mode.zip") -DestinationPath $sceneDir -ErrorAction Stop
}
```

用现有 `associate_frames` 产生真正配对的记录。对每个预选时刻，在前后 `0.15 s` 内取最近的一帧，不重复用同一文件，不用整张置信度为零的帧。这个上限只用于**挑选附近的已对齐 RGB 帧**，不是放宽对齐时的位姿误差门槛。命令打印参考时刻与实际选中帧，至少 6 张才复制到 `outputs/stage05_c_42446540_rgb`；无需新建 Python 源文件。

```powershell
conda activate yolov11
@'
from pathlib import Path
from shutil import copy2
import cv2 as cv
from rgbd_map.arkit_io import associate_frames, read_trajectory

scene = Path("data/arkitscenes/raw/Validation/42446540")
times, _ = read_trajectory(scene / "lowres_wide.traj")
records = associate_frames(scene, times)
print(f"associated_frames={len(records)}")
targets = (201472.994, 201473.294, 201475.393, 201499.999, 201500.499, 201501.998, 201506.996, 201508.996)
selected = []
for target in targets:
    nearby = sorted(
        (r for r in records if abs(r["timestamp"] - target) <= 0.15),
        key=lambda r: (abs(r["timestamp"] - target), r["timestamp"]),
    )
    for record in nearby:
        rgb = record["rgb_path"]
        if rgb in selected:
            continue
        confidence = cv.imread(str(record["confidence_path"]), cv.IMREAD_UNCHANGED)
        if confidence is None or not confidence.any():
            continue
        selected.append(rgb)
        print(f"{target:.3f} -> {rgb.name}, offset={record['timestamp']-target:+.3f}s")
        break
    else:
        print(f"{target:.3f} -> skipped: no eligible paired frame in window")
if len(selected) < 6:
    raise SystemExit("Fewer than six selected frames. Review the pairing before proceeding.")

output = Path("outputs/stage05_c_42446540_rgb")
output.mkdir(parents=True, exist_ok=True)
extras = {p.name for p in output.glob("*.png")} - {p.name for p in selected}
if extras:
    raise SystemExit(f"Existing RGB directory contains unexpected files: {sorted(extras)}")
for rgb in selected:
    copy2(rgb, output / rgb.name)
print(f"selected={len(selected)}, output={output}")
'@ | python -
```

先看实际复制的 RGB 确认保留了两把可区分的相似椅子和衣物遮挡证据。若配对不足或近邻帧关键线索消失，保留已知结果，在任何模型评分前核对选帧。通过后用固定提示词 `a chair. a dining table.` 和已有阈值运行：

```powershell
conda activate grounded_sam
python ../grounded_sam_project/predict.py --source outputs/stage05_c_42446540_rgb --text "a chair. a dining table." --box_threshold 0.35 --text_threshold 0.25 --output outputs/stage05_c_42446540_masks

conda activate yolov11
python -m scripts.extract_instances --scene data/arkitscenes/raw/Validation/42446540 --masks outputs/stage05_c_42446540_masks --output outputs/stage05_c_42446540_observations
python -m scripts.instance_map --observations outputs/stage05_c_42446540_observations/observations.json --min_voxel_iou 0.10 --output outputs/stage05_c_42446540_map
python -m scripts.export_instance_views --instances outputs/stage05_c_42446540_map/instances.json --detections outputs/stage05_c_42446540_masks/results.json --crop-mode box --output outputs/stage05_c_42446540_views
```

这四条命令分别生成掩码、世界坐标点云、地图 ID 和用于人工核验的视角裁剪。到这里先核对地图：若两把椅子被合为一个 ID，记录关联失败；若同一把椅子有重复 ID，按确认的实体指定多答案。不要先看检索分数再写查询目标。较笼统的“花纹餐椅”可能同时符合多个实体，不用于单答案难例。

#### 42446540 地图审查：同类难例成立，遮挡导致上游失败

实际关联 878 帧。从预定的 8 个参考时刻选到了 8 张同时具备有效配套数据的 RGB，分别为 `201473.044`、`201473.244`、`201475.443`、`201500.049`、`201500.549`、`201501.948`、`201507.046`、`201509.046`。每个时刻与原计划的偏差约 0.05 秒。8 张图共产生 11 条检测、11 条三维观测、11 张裁剪，关联后有 8 个原始地图 ID。下表在检索之前按掩码、裁剪和来源核对：

| 地图 ID | 观测数 | 图像及掩码证据 |
| --- | ---: | --- |
| 1 | 3 | 餐桌，框裁剪里显眼地包含了 ID 2 的椅子，但掩码主要落在两侧的桌面/桌布上。保留为干扰候选，不是椅子正例。 |
| 2 | 2 | 两个来源都看到同一把深色菱形花纹椅背的椅子，作为目标。 |
| 3 | 1 | 同一掩码里覆盖餐桌和披有黑白条纹夹克的椅子；没有找到该椅独立的候选。 |
| 4、6、8 | 各 1 | 同一餐桌的其他局部，原样保留。 |
| 5 | 1 | 掩码主要覆盖深色外套，衣物下的椅子缺少可验证的细节，不作本次检索正例。 |
| 7 | 1 | 相邻侧面的另一把餐椅，座面为深色叶纹，掩码边缘稍包含桌布，仍可凭画面辨认。 |

本次公开结果应区分两种情况：ID 2 与 ID 7 能用局部花纹属性区分，可用于**同类物体检索**；条纹夹克椅没有独立的 ID，是上游漏分/混合掩码，不能作为闭集检索失败来计算 Top-1 和 MRR。这里只能观察对相似椅子的排名变化，不能评价多视角是否恢复了被外套遮住的目标。只有 ID 1（3 个视角）和 ID 2（2 个视角）的聚合特征会改变；ID 7 在两组都只用同一帧。

由学习者新建 `experiments/fixtures/stage05_queries_42446540.json`。不改模型、候选集合、裁剪或评估函数。两个可辨认物体各写两条描述；以下文本和答案在推理之前固定：

```json
{
  "scene_id": "42446540",
  "queries": [
    {"query_id": "diamond_01", "text": "a dining chair seen from behind with a dark diamond-patterned back", "target_map_ids": [2]},
    {"query_id": "diamond_02", "text": "a tall gray chair back with black diamond motifs", "target_map_ids": [2]},
    {"query_id": "leaf_01", "text": "a side view of a dining chair with a black leaf-patterned seat", "target_map_ids": [7]},
    {"query_id": "leaf_02", "text": "a chair with a leaf-patterned cushion and curved metal legs", "target_map_ids": [7]}
  ]
}
```

从项目根目录运行。编码只做一次；全视角直接聚合，首视角先为每个 ID 选最早一帧再聚合。两份库都必须保留 8 个原始候选，包括错分和混合候选。最后两条用同一份查询表各写一组逐条报告：

```powershell
conda activate grounded_sam
python -m scripts.encode_instance_views --views outputs/stage05_c_42446540_views/views.json --output outputs/stage05_c_42446540_features
python -m scripts.aggregate_instance_features --features outputs/stage05_c_42446540_features/features.json --output outputs/stage05_c_42446540_multi_instances
python -m experiments.make_first_view_features --features outputs/stage05_c_42446540_features/features.json --output outputs/stage05_c_42446540_first_inputs
python -m scripts.aggregate_instance_features --features outputs/stage05_c_42446540_first_inputs/features.json --output outputs/stage05_c_42446540_first_instances
python -m scripts.evaluate_retrieval --features outputs/stage05_c_42446540_first_instances/instance_features.json --queries experiments/fixtures/stage05_queries_42446540.json --output outputs/stage05_c_42446540_eval_first
python -m scripts.evaluate_retrieval --features outputs/stage05_c_42446540_multi_instances/instance_features.json --queries experiments/fixtures/stage05_queries_42446540.json --output outputs/stage05_c_42446540_eval_multi
```

验收：两库都是 `(8, 768)`，两组报告各有 4 条（2 个真实目标）。对照每条的 `target_rank` 和 `top1_map_id`，区分「把 A 当成 B」和「被包含相同画面内容的餐桌框吸引」两类错误；遮挡椅的漏分单独记作上游失败。仅当实测排名变化才报告检索差异。

#### 42446540 对照实测

核对了学习者落盘的两组报告与固定查询表。编码矩阵为 `(11, 768)`，两组候选矩阵均为 `(8, 768)`；按源索引复算，首视角确为每个 ID 的最早帧，两个条件中只有 ID 1 与 ID 2 的特征变化。两组 Top-1 都是 4/4、MRR 都是 1.0，四条正确目标的名次均为 1，预测第一名 ID 均为 `[2, 2, 7, 7]`。两条 ID 2 查询的首位相似度有小幅变化，不对应排名收益，也无法从保存的报告推断与第二名的分差。三场景小规模验收及遮挡失败案例见[报告](../reports/stage05_retrieval_comparison.md#第二个留出场景42446540)。
