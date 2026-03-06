"""
RTSP 视频流检测服务
从 RTSP 视频流中定期提取帧，进行裁剪和 DINO 物体识别
后台常驻服务，命令行输出日志
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
import logging
from datetime import datetime
import threading
import signal
import sys
import subprocess
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 打印 OpenCV 信息
logger.info(f"OpenCV 版本: {cv2.__version__}")
build_info = cv2.getBuildInformation()
if build_info:
    first_line = build_info.split('\n')[0]
    logger.info(f"OpenCV 后端: {first_line}")
else:
    logger.info("OpenCV 后端: N/A")


# ========== 配置参数 ==========
CONFIG = {
    # RTSP 配置
    "rtsp_url": "rtsp://127.0.0.1:19345/video",

    # 裁剪参数
    "crop_width": 550,
    "crop_height": 870,
    "crop_x": 910,
    "crop_y": 530,
    "enable_crop": True,

    # 检测参数
    "detection_caption": "bike",
    "box_threshold": 0.12,
    "text_threshold": 1.0,

    # 框大小筛选
    "max_box_area": 6000,
    "min_box_area": 150,

    # 采样间隔（秒）
    "sample_interval": 10,

    # 输出目录
    "output_dir": "output/rtsp_detection",

    # 临时图片路径
    "temp_frame_path": "temp_rtsp_frame.jpg",

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ========== 全局变量 ==========
running = True
frame_count = 0
detection_count = 0
total_detected_items = 0


# ========== 信号处理 ==========
def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global running
    logger.info("收到停止信号，正在关闭服务...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)


# ========== 资源监控 ==========
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self.peak_ram_mb = 0
        self.peak_gpu_mb = 0
        self.lock = threading.Lock()

    def get_ram_usage_mb(self):
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0

    def get_gpu_usage_mb(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def update_peak(self):
        with self.lock:
            ram = self.get_ram_usage_mb()
            gpu = self.get_gpu_usage_mb()
            if ram > self.peak_ram_mb:
                self.peak_ram_mb = ram
            if gpu > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu

    def get_summary(self):
        with self.lock:
            return f"RAM: {self.peak_ram_mb:.0f} MB, GPU: {self.peak_gpu_mb:.0f} MB"


resource_monitor = ResourceMonitor()


# ========== 模型加载 ==========
def load_groundingdino_model():
    """加载 GroundingDINO 模型"""
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    project_root = Path(__file__).resolve().parents[0]
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = project_root / "weights" / "groundingdino_swint_ogc.pth"

    logger.info(f"加载 GroundingDINO 模型...")
    logger.info(f"  配置文件: {config_file}")
    logger.info(f"  权重文件: {checkpoint_path}")

    args = SLConfig.fromfile(config_file)
    args.device = CONFIG["device"]
    model = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model = model.to(CONFIG["device"])
    model.eval()

    logger.info(f"✓ 模型加载完成 (设备: {CONFIG['device']})")
    return model


# ========== 图像变换 ==========
def image_transform_grounding(init_image):
    import groundingdino.datasets.transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None)
    return init_image, image


# ========== 裁剪图像 ==========
def crop_image(image_np, crop_width, crop_height, crop_x, crop_y):
    """裁剪图像"""
    h, w = image_np.shape[:2]

    # 确保裁剪区域在图像范围内
    crop_x = min(crop_x, w - crop_width)
    crop_y = min(crop_y, h - crop_height)

    cropped = image_np[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    return cropped


# ========== 检测函数 ==========
def detect_objects(model, image_np, caption, box_threshold, text_threshold,
                   max_box_area, min_box_area):
    """检测图像中的物体"""
    from groundingdino.util.inference import predict

    # 转换为 PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGB")

    # 图像变换
    _, image_tensor = image_transform_grounding(image_pil)

    # 运行检测
    boxes, logits, phrases = predict(
        model,
        image_tensor,
        caption,
        box_threshold,
        text_threshold,
        device=CONFIG["device"]
    )

    # 按框大小筛选
    filtered_results = []

    if boxes is not None and len(boxes) > 0:
        img_h, img_w = image_np.shape[:2]

        for box, logit, phrase in zip(boxes, logits, phrases):
            # box格式: [cx, cy, w, h] (归一化坐标)
            cx, cy, bw, bh = box

            # 计算实际像素尺寸
            box_w = bw * img_w
            box_h = bh * img_h
            box_area = box_w * box_h

            # 计算实际像素坐标
            x1 = cx * img_w - box_w / 2
            y1 = cy * img_h - box_h / 2
            x2 = cx * img_w + box_w / 2
            y2 = cy * img_h + box_h / 2

            # 筛选条件
            if max_box_area > 0 and box_area > max_box_area:
                continue
            if min_box_area > 0 and box_area < min_box_area:
                continue

            filtered_results.append({
                "label": phrase,
                "confidence": float(logit),
                "center_x": float(cx),
                "center_y": float(cy),
                "width": float(box_w),
                "height": float(box_h),
                "area": float(box_area),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })

    return filtered_results


# ========== 保存结果 ==========
def save_detection_result(frame_np, cropped_np, detections, output_dir, timestamp):
    """保存检测结果（带识别框）"""
    from groundingdino.util.inference import annotate
    import groundingdino.datasets.transforms as T

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    filename = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    # 保存原始帧
    original_path = output_dir / "original" / f"{filename}.jpg"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(original_path), frame_np)

    # 保存裁剪后的帧（无标注）
    cropped_path = output_dir / "cropped" / f"{filename}.jpg"
    cropped_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cropped_path), cropped_np)

    # 创建带标注的图像
    if detections:
        # 将检测结果转换为 GroundingDINO 格式
        boxes_list = []
        logits_list = []
        phrases_list = []

        for det in detections:
            # 转换为归一化坐标 [cx, cy, w, h]
            h_img, w_img = cropped_np.shape[:2]
            cx = det["center_x"]
            cy = det["center_y"]
            bw = det["width"] / w_img
            bh = det["height"] / h_img

            boxes_list.append([cx, cy, bw, bh])
            logits_list.append(det["confidence"])
            phrases_list.append(det["label"])

        if boxes_list:
            import torch
            boxes_tensor = torch.tensor(boxes_list)
            logits_tensor = torch.tensor(logits_list)

            # 使用 annotate 函数绘制
            annotated = annotate(
                image_source=cropped_np,
                boxes=boxes_tensor,
                logits=logits_tensor,
                phrases=phrases_list
            )

            # 转换颜色空间
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # 保存带标注的图像
            annotated_path = output_dir / "annotated" / f"{filename}.jpg"
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotated_path), annotated_rgb)
        else:
            annotated_path = None
    else:
        # 无检测结果，复制裁剪图
        annotated_path = output_dir / "annotated" / f"{filename}.jpg"
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), cropped_np)

    # 保存检测结果 JSON
    result = {
        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "timestamp_unix": timestamp.timestamp(),
        "original_image": str(original_path),
        "cropped_image": str(cropped_path),
        "annotated_image": str(annotated_path) if annotated_path else None,
        "detection_count": len(detections),
        "detections": detections
    }

    json_path = output_dir / "json" / f"{filename}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return str(json_path)


# ========== 更新汇总 JSON ==========
def update_summary_json(output_dir, latest_result):
    """更新汇总 JSON 文件"""
    output_dir = Path(output_dir)
    summary_file = output_dir / "summary.json"

    # 读取现有摘要
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = {
            "service_start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "rtsp_url": CONFIG["rtsp_url"],
            "total_frames_processed": 0,
            "total_detections": 0,
            "latest_detections": []
        }

    # 更新摘要
    summary["total_frames_processed"] = frame_count
    summary["total_detections"] = total_detected_items
    summary["last_update_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary["resource_usage"] = resource_monitor.get_summary()

    # 添加最新检测（保留最近100条）
    summary["latest_detections"].insert(0, latest_result)
    summary["latest_detections"] = summary["latest_detections"][:100]

    # 保存
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ========== 主循环 ==========
def main():
    global running, frame_count, detection_count, total_detected_items

    logger.info("=" * 60)
    logger.info("RTSP 视频流检测服务")
    logger.info("=" * 60)

    # 打印配置
    logger.info("配置参数:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 60)

    # 加载模型
    try:
        model = load_groundingdino_model()
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 创建输出目录
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_file = output_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    # 使用 ffmpeg 从 RTSP 流获取帧（直接保存到文件）
    def get_frame_from_rtsp_ffmpeg(rtsp_url, output_path, timeout=15):
        """使用 ffmpeg 从 RTSP 流获取一帧并保存到文件"""
        cmd = [
            'ffmpeg',
            '-y',                                # 覆盖输出文件
            '-rtsp_transport', 'tcp',              # 使用 TCP 传输
            '-i', rtsp_url,
            '-vframes', '1',                        # 只读取1帧
            '-q:v', '2',                            # 高质量
            '-update', '1',                         # 更新模式（覆盖文件）
            '-nostats',                            # 不显示统计信息
            '-loglevel', 'error',                   # 只显示错误
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            if result.returncode != 0:
                logger.debug(f"ffmpeg 错误: {result.stderr}")
                return False

            # 检查文件是否创建成功
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True

            return False

        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg 超时")
            return False
        except Exception as e:
            logger.warning(f"ffmpeg 执行失败: {e}")
            return False

    def load_frame_from_file(file_path):
        """从文件加载帧"""
        if not os.path.exists(file_path):
            return None

        frame = cv2.imread(file_path)
        if frame is None:
            return None

        return frame

    logger.info("使用 ffmpeg 保存帧到文件，然后读取")

    # 测试 ffmpeg 连接
    logger.info("测试 RTSP 连接...")
    temp_path = CONFIG["temp_frame_path"]

    # 清理可能存在的旧临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 测试获取一帧
    if not get_frame_from_rtsp_ffmpeg(CONFIG["rtsp_url"], temp_path):
        logger.error("无法从 RTSP 流获取帧")
        logger.error("请确保:")
        logger.error("  1. RTSP 服务器正在运行")
        logger.error(f"  2. URL 正确: {CONFIG['rtsp_url']}")
        logger.error(f"  3. 可以手动测试: ffmpeg -rtsp_transport tcp -i {CONFIG['rtsp_url']} -vframes 1 test.jpg")
        return

    # 读取测试帧
    test_frame = load_frame_from_file(temp_path)
    if test_frame is None:
        logger.error(f"无法读取保存的图片: {temp_path}")
        return

    h, w = test_frame.shape[:2]
    logger.info(f"✓ RTSP 连接成功!")
    logger.info(f"  分辨率: {w}x{h}")
    logger.info(f"  采样间隔: {CONFIG['sample_interval']} 秒")
    logger.info("=" * 60)

    last_sample_time = 0
    consecutive_empty_frames = 0

    try:
        while running:
            current_time = time.time()

            # 检查是否需要采样
            if current_time - last_sample_time < CONFIG["sample_interval"]:
                time.sleep(0.5)
                continue

            # 更新资源监控
            resource_monitor.update_peak()

            # 采样时间到，获取新帧
            last_sample_time = current_time
            frame_count += 1
            timestamp = datetime.now()

            logger.info(f"[帧 #{frame_count}] 开始处理 ({timestamp.strftime('%H:%M:%S')})")

            # 使用 ffmpeg 获取帧并保存到临时文件
            temp_path = CONFIG["temp_frame_path"]
            if not get_frame_from_rtsp_ffmpeg(CONFIG["rtsp_url"], temp_path):
                logger.warning("无法读取帧，等待后重试...")
                consecutive_empty_frames += 1

                if consecutive_empty_frames > 5:
                    logger.error("连续读取失败，等待 30 秒后重试...")
                    time.sleep(30)
                    consecutive_empty_frames = 0
                else:
                    time.sleep(2)

                continue

            consecutive_empty_frames = 0

            # 从文件读取帧
            frame = load_frame_from_file(temp_path)
            if frame is None:
                logger.error(f"无法读取保存的图片: {temp_path}")
                continue

            # 检查帧质量（检测灰色/空帧）
            gray_mean = np.mean(frame)
            gray_std = np.std(frame)

            if gray_std < 10:
                logger.warning(f"检测到低质量帧 (均值={gray_mean:.1f}, 标准差={gray_std:.1f})，跳过")
                continue

            logger.debug(f"帧质量正常 (均值={gray_mean:.1f}, 标准差={gray_std:.1f})")

            try:
                # 裁剪图像
                if CONFIG["enable_crop"]:
                    cropped = crop_image(
                        frame,
                        CONFIG["crop_width"],
                        CONFIG["crop_height"],
                        CONFIG["crop_x"],
                        CONFIG["crop_y"]
                    )
                    detect_image = cropped
                else:
                    detect_image = frame

                # 检测物体
                detections = detect_objects(
                    model,
                    detect_image,
                    CONFIG["detection_caption"],
                    CONFIG["box_threshold"],
                    CONFIG["text_threshold"],
                    CONFIG["max_box_area"],
                    CONFIG["min_box_area"]
                )

                detection_count += 1
                total_detected_items += len(detections)

                # 保存结果
                if CONFIG["enable_crop"]:
                    save_frame = cropped
                else:
                    save_frame = frame

                json_path = save_detection_result(
                    frame,
                    save_frame,
                    detections,
                    CONFIG["output_dir"],
                    timestamp
                )

                # 更新汇总
                with open(json_path, 'r', encoding='utf-8') as f:
                    latest_result = json.load(f)
                update_summary_json(CONFIG["output_dir"], latest_result)

                # 输出结果
                if detections:
                    logger.info(f"  检测到 {len(detections)} 个 {CONFIG['detection_caption']}")
                    for i, det in enumerate(detections):
                        logger.info(f"    [{i+1}] {det['label']} - 置信度: {det['confidence']:.3f}, "
                                  f"面积: {det['area']:.0f} px²")
                else:
                    logger.info(f"  未检测到 {CONFIG['detection_caption']}")

                logger.info(f"  结果已保存: {json_path}")
                logger.info(f"  资源: {resource_monitor.get_summary()}")

            except Exception as e:
                logger.error(f"处理失败: {e}")
                import traceback
                traceback.print_exc()

            logger.info("-" * 40)

    except KeyboardInterrupt:
        pass

    finally:
        logger.info("=" * 60)
        logger.info("服务统计:")
        logger.info(f"  总帧数: {frame_count}")
        logger.info(f"  检测次数: {detection_count}")
        logger.info(f"  总检测数: {total_detected_items}")
        logger.info(f"  资源峰值: {resource_monitor.get_summary()}")
        logger.info("=" * 60)
        logger.info("服务已停止")


if __name__ == "__main__":
    main()
