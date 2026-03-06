# RTSP 视频流检测服务

从 RTSP 视频流中定期提取帧，进行裁剪和 DINO 物体识别的后台服务。

## 功能特点

- 从 RTSP 视频流实时采样
- 按设定间隔（默认60秒）提取一帧
- 自动裁剪图像
- GroundingDINO 物体检测
- 框大小筛选（过滤大物体和噪声）
- 保存检测结果和图像
- 实时日志输出
- 资源使用监控

## 使用方法

### 1. 启动 RTSP 服务器

首先确保 RTSP 视频流正在运行：

```bash
python start_rtsp_server.py
```

### 2. 启动检测服务

**方式一：使用批处理文件**
```
start_rtsp_detection.bat
```

**方式二：命令行**
```bash
python rtsp_detection_service.py
```

### 3. 停止服务

按 `Ctrl+C` 停止服务

## 配置参数

在 `rtsp_detection_service.py` 中修改 `CONFIG` 字典：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `rtsp_url` | RTSP 视频流地址 | rtsp://127.0.0.1:19345/video |
| `crop_width` | 裁剪宽度 | 550 |
| `crop_height` | 裁剪高度 | 870 |
| `crop_x` | 裁剪起始X | 910 |
| `crop_y` | 裁剪起始Y | 530 |
| `enable_crop` | 是否启用裁剪 | True |
| `detection_caption` | 检测提示词 | "bike" |
| `box_threshold` | Box阈值 | 0.12 |
| `text_threshold` | Text阈值 | 1.0 |
| `max_box_area` | 最大框面积（像素²） | 6000 |
| `min_box_area` | 最小框面积（像素²） | 150 |
| `sample_interval` | 采样间隔（秒） | 60 |
| `output_dir` | 输出目录 | output/rtsp_detection |

## 输出结果

### 目录结构
```
output/rtsp_detection/
├── original/          # 原始帧
├── cropped/           # 裁剪后的帧
├── json/              # 检测结果JSON
├── config.json        # 当前配置
└── summary.json       # 汇总统计
```

### JSON 格式

**单帧检测结果 (json/frame_YYYYMMDD_HHMMSS.json):**
```json
{
  "timestamp": "2026-03-06 14:30:00",
  "timestamp_unix": 1709704200.123,
  "original_image": "output/rtsp_detection/original/frame_20260306_143000.jpg",
  "cropped_image": "output/rtsp_detection/cropped/frame_20260306_143000.jpg",
  "detection_count": 3,
  "detections": [
    {
      "label": "bike",
      "confidence": 0.856,
      "center_x": 0.512,
      "center_y": 0.487,
      "width": 45.2,
      "height": 78.5,
      "area": 3550,
      "x1": 489,
      "y1": 447,
      "x2": 535,
      "y2": 526
    }
  ]
}
```

**汇总统计 (summary.json):**
```json
{
  "service_start_time": "2026-03-06 14:00:00",
  "rtsp_url": "rtsp://127.0.0.1:19345/video",
  "total_frames_processed": 120,
  "total_detections": 245,
  "last_update_time": "2026-03-06 16:30:00",
  "resource_usage": "RAM: 2048 MB, GPU: 1536 MB",
  "latest_detections": [...]
}
```

## 日志输出示例

```
2026-03-06 14:30:00 [INFO] ============================================================
2026-03-06 14:30:00 [INFO] RTSP 视频流检测服务
2026-03-06 14:30:00 [INFO] ============================================================
2026-03-06 14:30:00 [INFO] 配置参数:
2026-03-06 14:30:00 [INFO]   rtsp_url: rtsp://127.0.0.1:19345/video
2026-03-06 14:30:00 [INFO]   sample_interval: 60
...
2026-03-06 14:31:00 [INFO] [帧 #1] 开始处理 (14:31:00)
2026-03-06 14:31:02 [INFO]   检测到 3 个 bike
2026-03-06 14:31:02 [INFO]     [1] bike - 置信度: 0.856, 面积: 3550 px²
2026-03-06 14:31:02 [INFO]     [2] bike - 置信度: 0.723, 面积: 2890 px²
2026-03-06 14:31:02 [INFO]     [3] bike - 置信度: 0.689, 面积: 4200 px²
2026-03-06 14:31:02 [INFO]   结果已保存: output/rtsp_detection/json/frame_20260306_143100.json
2026-03-06 14:31:02 [INFO]   资源: RAM: 1890 MB, GPU: 1234 MB
```

## 常见问题

### 无法连接到 RTSP 流
1. 检查 RTSP 服务器是否正在运行
2. 确认 URL 正确
3. 检查防火墙设置

### 检测结果为空
1. 调整 `detection_caption`（检测提示词）
2. 降低 `box_threshold` 和 `text_threshold`
3. 检查 `max_box_area` 和 `min_box_area` 设置
4. 确认裁剪参数正确

### 内存不足
1. 增大 `sample_interval`（减少采样频率）
2. 降低检测分辨率
3. 使用 CPU 而非 GPU

## 后台运行

**Windows (使用 start):**
```bash
start /B python rtsp_detection_service.py > service.log 2>&1
```

**Windows (使用 pyinstaller 打包后):**
```bash
rtsp_detection_service.exe
```

## 依赖

- Python 3.7+
- PyTorch
- GroundingDINO
- OpenCV
- NumPy
- PIL
- psutil (可选，用于资源监控)

## 许可

基于 GroundingDINO，遵循原项目许可。
