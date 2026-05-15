# 开发指南（开发者必读）

本文件包含项目的分阶段计划、YOLO 数据集准备与训练指南、ByteTrack 设计理由、Kalman/Savitzky-Golay 说明、小目标检测优化与性能优化建议。

## 分阶段计划

Phase 1: ROI 与颜色检测
- 目标: 实现稳定的 HSV 颜色检测器并产出 bbox + center + team_color
- 输入: 比赛录像、参考小地图
- 输出: 初步 bbox 与轨迹草图
- Debug: 可视化 mask、轮廓
- 常见 Bug: HSV 阈值不严谨、形态学核太大/太小
- 验证: 手工标注若干帧，对比检测召回/精度
- 性能: 使用下采样与 ROI 提速

Phase 2: YOLO 检测与数据集训练
- 目标: 训练 `player_marker` 类别的 YOLOv11n 权重以增强鲁棒性
- 输入: 抽帧、标注后的 images/labels
- 输出: best.pt
- Debug: 可视化预测 bbox、conf
- 验证: mAP、召回率曲线
- 性能: 使用 GPU、AMP、合理 batch、imgsiz

Phase 3: ByteTrack 多目标跟踪
- 目标: 实现轻量 ByteTrack 风格关联模块（Hungarian + 距离矩阵）
- 验证: ID switch、track fragmentation 统计

Phase 4: Kalman + Savitzky 平滑
- 目标: 实时 Kalman 平滑，最终轨迹使用 Savitzky-Golay 进行美化
- 验证: 轨迹噪声下降、速度稳定

Phase 5: Homography 坐标映射
- 目标: 支持 4 点手动标定与 homography 保存/加载
- 验证: 将像素坐标映射至地图坐标，检查误差

Phase 6: 科研级可视化
- 目标: 生成 `trajectory.png`（渐变色轨迹、起终点、图例）与 `debug.mp4`

## YOLO 数据集制作与训练

数据集结构:

datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/

labels 使用 YOLO 格式：每行 `class x_center y_center w h`（相对坐标）。

示例 dataset.yaml:

```yaml
path: datasets
train: images/train
val: images/val
names:
  0: player_marker
``` 

抽帧与标注:
1. 使用 `ffmpeg -i match.mp4 -vf fps=5 output_%06d.jpg` 抽帧
2. 使用 LabelImg/Roboflow/VoTT 标注，导出 YOLO 格式

训练命令（ultralytics）：

```bash
# 使用 ultralytics CLI（yolo）
yolo detect train data=datasets/dataset.yaml model=yolov11n.pt imgsz=640 batch=16 epochs=100 device=0
```

导出 best.pt: ultralytics 会在 `runs/detect/train/weights/best.pt` 生成

推理命令:
```bash
yolo detect predict model=best.pt source=tests/images/img1.jpg imgsz=640 conf=0.25
```

小目标优化策略详见下节。

## 小目标检测优化（地图上的玩家标记）

1. imgsz 选择：增大 `imgsz` 提升小目标可见性，但增加显存与推理延迟。常用 640→1024，视显存而定。
2. Mosaic：保留 mosaic 可以提高对小目标的鲁棒性，但对密集小目标有时造成遮挡误导。
3. Copy-Paste：增强小目标样本密度，尤其对稀疏颜色标记有效。
4. Anchor/AutoAnchor：使用 k-means 计算适配小目标的 anchors 或使用 anchor-free 模型。
5. conf threshold：适当降低 `conf`（例如 0.2）增加召回，再由跟踪器过滤假阳性。
6. 为什么会漏检：目标尺寸太小、形态多样、相邻遮挡、motion blur。解决方法：更高分辨率输入、增强、增加标注样本。

## ByteTrack 适配理由与实现要点

为什么 ByteTrack 适合地图跟踪：
- 地图目标通常是小、稠密且持续移动，ByteTrack 的两个阶段（高置信度先匹配、低置信度后续匹配）能提升短时断连容忍性。
- 使用卡尔曼预测 + Hungarian 匹配，能稳定 ID 分配并降低 ID switch。

实现要点：
- 构建距离矩阵（中心点欧式距离）并使用 Hungarian 求最优分配
- 匹配级联（可按置信度分层）
- 对未匹配检测初始化新轨迹，未匹配轨迹计数 lost_frames，超过阈值删除
- 速度估计使用中心差分（也可以用卡尔曼状态量）

## Kalman vs Savitzky-Golay

- Kalman：适合实时，计算复杂度低，能逐帧预测与更新，抗噪声且能估计速度（适合在线平滑）
- Savitzky-Golay：基于窗口的多项式拟合，适合离线后处理（最终轨迹美化），保持信号形态并去除噪点

组合：实时使用 Kalman 提供稳定输出，轨迹收集完成后用 Savitzky-Golay 做最终平滑。

## 性能优化建议

1. GPU 推理：使用半精度推理（AMP）、批量推理、预加载模型到 GPU
2. OpenCV IO：使用 `cv2.CAP_PROP_POS_FRAMES` 跳帧、解码线程优化、避免逐帧写磁盘
3. Batch inference：把多帧合批到模型（若显存允许）以减少调用开销
4. Tracker 优化：使用 numpy 向量化距离计算、限制候选匹配范围
5. 轨迹缓存：使用内存缓存并定期持久化为 parquet/csv
6. Debug 编码：选用 `ffmpeg` 的硬件编码（如果可用），降低 CPU 负担

## 调试与验证

- 可视化中间产物（mask、contours、yolo bboxes）
- 保存示例帧与注释用于回归测试
- 统计指标：ID switches、MOTA（可选）

## 运行说明

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行：

```bash
python -m src.pipeline --config configs/default.yaml
```

## 后续工作建议

- 完善 hybrid detector 的置信度融合策略
- 支持多人同时标注的训练数据管线
- 添加可视化 Web 前端
- 更精细的速度估计（使用差分时间 dt）

