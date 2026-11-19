# 第三关：ARKitScenes 单场景数据检查

日期：2026-11-15；进度更新：2026-11-16。数据下载与本页预检查已完成。学习者随后完成数据适配与两段 10 帧重建检查，见 [A 实测记录](stage03_arkit_preview.md)。本页保留最初的数据准备记录。

## 来源与范围

使用 [ARKitScenes 官方文档](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md) 下载示例中的 Training 场景 `47333462`。按 [官方下载脚本](https://github.com/apple-aiml-research/ARKitScenes/blob/main/download_data.py) 提供的 Apple 地址，仅下载所需单场景资源：

`https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/47333462/`

| 文件 | 下载字节数 |
|---|---:|
| lowres_wide.zip | 268393149 |
| lowres_depth.zip | 88437780 |
| lowres_wide_intrinsics.zip | 738552 |
| confidence.zip | 3560346 |
| lowres_wide.traj | 59655 |
| 47333462_3dod_annotation.json | 9890 |

总计 361199372 字节，约 344.47 MiB。已核对响应头、下载文件大小，四个 ZIP 均通过内部 CRC 检查；解压路径限定在该场景目录内。ZIP 原件保留。

本地目录：`E:\shixi\rgbd_instance_map\data\arkitscenes\raw\Training\47333462`。该目录位于项目已忽略的 data 下。

## 实际文件与关联检查

| 内容 | 数量 |
|---|---:|
| RGB PNG | 3611 |
| 深度 PNG | 3610 |
| 内参 .pincam | 3611 |
| 置信度 PNG | 3610 |
| 轨迹记录 | 596 |
| 文件齐全且最近轨迹时间差小于 5 ms 的帧 | 596 |

以深度为基准，RGB、置信度和同名内参均可配齐。其余 3014 张深度图因最近轨迹时间差超过本次门槛被跳过；这反映图像与轨迹的采样率不同，不表示下载丢失。

596 组的位姿时间差最大约 0.000499920 秒，平均约 0.000252383 秒。此关联统计是数据准备检查，尚不代表学习者实现已通过验收。

## 首帧样例

文件主名：`47333462_57352.888`。最近轨迹索引为 0，轨迹时间为 57352.88780654 秒，差约 0.000193460 秒。

- RGB：shape=(192,256,3)，dtype=uint8。
- 深度：shape=(192,256)，dtype=uint16。
- 置信度：shape=(192,256)，首帧实际值为 1、2。
- 非零深度中位数：0.58 米，按毫米除以 1000 换算。

首个内参文件的六个值为：

```text
256 192 212.811 212.811 126.594 93.1247
```

它们依次为 width、height、fx、fy、cx、cy。

首条轨迹为：

```text
57352.88780654 -2.217554497473789 2.1758070205829703 -0.005041953551818877 0.0589587 0.046871 0.0094436
```

旋转为轴角向量，布局遵循 timestamp、rx、ry、rz、tx、ty、tz。按官方加载器构造世界到相机矩阵后取逆，以获得后续需要的 T_wc。

## 标注概览及用途

官方场景标注包含 cabinet 3 个、bed 1 个、table 1 个、tv_monitor 2 个、chair 2 个。这是场景标注统计，不是模型预测数量，也不保证同一张图片能看见全部物体。

这些标注留作后续检查或评价，不将其中的 UID 或框作为预测阶段的实例关联答案。Grounding DINO 与 SAM 2 的图像推理尚未在本场景执行。
