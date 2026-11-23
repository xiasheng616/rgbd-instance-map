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

学习者已完成两组评估。逐项核对到 15 条视角特征、相同的 8 个原始地图 ID、相同的 8 条查询与多答案标注；两个实例矩阵均为 `(8, 768)`，按源索引复算与落盘结果一致。两组 Top-1 都是 8/8，MRR 都是 1.0，没有查询改变第一个正确结果的名次。查询 `fridge_02` 的第一名 ID 从 5 换到 6，但两者都是事先核定的同一台冰箱。[逐条结果与双场景报告](../reports/stage05_retrieval_comparison.md)
