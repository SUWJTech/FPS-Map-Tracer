# FPS-Map-Tracer

FPS 小地图轨迹分析与全局地图可视化系统。支持：

- 小地图目标检测（YOLO）
- 多目标跟踪（ByteTrack-lite + Kalman）
- 轨迹导出（CSV/JSON）
- 小地图到大地图坐标映射（Homography）
- 轨迹渲染、热力图、事件标注
- Streamlit WebUI 一键流程（上传视频 -> 自动跑完整流程 -> 可视化）

---

## 1. 项目结构

```text
fps-trajectory/
├── configs/
│   ├── default.yaml
│   ├── map.jpg
│   └── homography_matrix.json
├── src/
│   ├── pipeline.py
│   ├── tracker.py
│   ├── trajectory_manager.py
│   ├── map_registration.py
│   ├── professional_renderer.py
│   ├── advanced_heatmap.py
│   └── speed_heatmap.py
├── map_mapper.py
├── render_map.py
├── webui.py
├── requirements.txt
└── outputs/
```

---

## 2. 环境安装

建议 Python 3.10~3.12（3.13 在部分三方包下可能有兼容问题）。

```bash
cd fps-trajectory
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果你用 Conda：

```bash
conda create -n fpsmap python=3.11 -y
conda activate fpsmap
pip install -r requirements.txt
```

---

## 3. 快速开始（命令行方式，稳定推荐）

这是当前验证过效果最稳定的流程：

```bash
python -u -m src.pipeline --config configs/default.yaml
python map_mapper.py --config configs/default.yaml --trajectory_csv outputs/trajectory.csv --map_image configs/map.jpg --matrix configs/homography_matrix.json --trajectory_json outputs/trajectory.json --output_csv outputs/mapped_trajectory.csv --debug_match_png outputs/registration_matches.png --force_reregister
python render_map.py --mapped_csv outputs/mapped_trajectory.csv --map_image configs/map.jpg --output_dir outputs --team all --events_csv outputs/events.csv --events_json outputs/events.json --trajectory_json outputs/trajectory.json --frame_start 0 --frame_end -1
```

生成文件主要在 `outputs/`：

- `trajectory.csv`, `trajectory.json`
- `mapped_trajectory.csv`
- `final_trajectory.png`
- `all_team_routes.png` / `team_x_team_routes.png`
- `heatmap.png`, `density_heatmap.png`, `speed_heatmap.png`, `stay_heatmap.png`
- `event_markers.png`

---

## 4. WebUI 使用（上传视频 + 一键全流程）

```bash
streamlit run webui.py
```

### WebUI 推荐操作顺序

1. 上传视频（左侧 Upload）
2. 设置 `Output run name`
3. 选择队伍过滤（`all` 或 `team_x`）
4. 点击 `🚀 One-click Full Pipeline (1→2→4)`

### Compatibility mode 说明

- 开启后：严格使用 `configs/default.yaml + outputs/`
- 关闭后：使用上传视频 + 独立输出目录（推荐）

如果你发现“上传了新视频但结果和旧视频一样”，通常是 Compatibility mode 开启导致读取了 `default.yaml` 的 `video_path`。

---

## 5. 数据字段说明

### trajectory.csv

- `frame_id`, `timestamp`, `track_id`
- `x`, `y`（小地图坐标）
- `vx`, `vy`, `confidence`, `predicted`
- `team_color`（兼容字段，可能为 unknown）
- `final_team`（推荐使用）
- `team`（`final_team` 别名）

### mapped_trajectory.csv

在 `trajectory.csv` 基础上增加：

- `map_x`, `map_y`（大地图坐标）
- `final_team`, `team`

> 注意：当前系统以 `final_team/team` 作为正式队伍字段；`team_color` 仅保留兼容。

---

## 6. 地图坐标映射原理

使用 Homography（单应性变换）：

1. ORB 提取小地图/大地图特征
2. BFMatcher 匹配 + ratio test
3. RANSAC 估计 3x3 单应矩阵 `H`
4. 用 `cv2.perspectiveTransform` 将 `(x, y)` 映射为 `(map_x, map_y)`

---

## 7. 常见问题（FAQ）

### Q1: team_color 都是 unknown，是不是坏了？
不是。当前架构是“轨迹后验聚类”，应使用 `final_team/team`。

### Q2: 一键流程效果和手工命令不一致？
确认：
- Compatibility mode 状态
- `Effective video_path`
- 一键渲染是否使用全时段（`frame_start=0`, `frame_end=-1`）

---

## 8. 处理结果和技术报告

- 数据集：fps-trajectory/artifacts/yolov11-ASF-P2/data/dataset.zip
- 处理结果：fps-trajectory/outputs
- 技术报告：fps-trajectory/系统技术报告.docx

<div align="center">
  <div style="display: flex; gap: 8px; overflow-x: auto; padding: 10px 0;">
    <img src="https://github.com/user-attachments/assets/74f94fc3-8056-438a-8f17-86048c275745" alt="EduMeta Preview 1" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github.com/user-attachments/assets/f2a9e58a-64a8-4bdc-b46b-9bc957b0daa1" alt="EduMeta Preview 2" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github.com/user-attachments/assets/0bffe488-a5e4-407e-8b0f-1eb1281f7b2e" alt="EduMeta Preview 3" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github.com/user-attachments/assets/91a38977-7030-423b-a60f-bef9271cc8e9" alt="EduMeta Preview 4" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github.com/user-attachments/assets/8f2451f6-7947-4b8a-bf46-64914a0c3e07" alt="EduMeta Preview 5" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github.com/user-attachments/assets/24c6d309-fe8e-4b03-9aba-e1611ca48bf8" alt="EduMeta Preview 6" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github.com/user-attachments/assets/a87cf565-431d-483a-b754-7ed48dd081ca" alt="EduMeta Preview 7" style="height: 360px; border-radius: 10px; object-fit: cover;">
    <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/229098271/573377327-ed592ce7-5892-49bb-ad4a-a147c953e5dd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260403%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260403T035848Z&X-Amz-Expires=300&X-Amz-Signature=d51103a89bafad6f3309465d7b97804a752b2fa5bfbbe0e9bbd4a6a763a5b8cf&X-Amz-SignedHeaders=host" alt="EduMeta Preview 8" style="height: 360px; border-radius: 10px; object-fit: cover;">
  </div>
</div>
---

