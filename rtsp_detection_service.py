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
from typing import Optional, Dict, List, Tuple

from is3_metadata_api import MetadataAPI

# 导入图像配准模块
try:
    from image_registration import ImageRegistration, OffsetSmoother, DinoTreeGridRegistration
    IMAGE_REGISTRATION_AVAILABLE = True
except ImportError:
    IMAGE_REGISTRATION_AVAILABLE = False
    DinoTreeGridRegistration = None
    logging.warning("图像配准模块不可用，多区域偏移补偿功能将被禁用")

# 检查scipy是否可用（用于网格配准）
try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None
    logging.info("scipy未安装，网格配准功能不可用")

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
    "rtsp_url": "",

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
    "sample_interval": 30,

    # 输出目录
    "output_dir": "output/rtsp_detection",

    # 临时图片路径
    "temp_frame_path": "temp_rtsp_frame.jpg",

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # iS3 上报配置
    "is3": {
        "enabled": True,
        "base_url": "https://server.is3.net.cn",
        "file_base_url": "https://file.is3.net.cn",
        "project_id": os.getenv("IS3_PROJECT_ID", ""),
        "access_key": os.getenv("IS3_ACCESS_KEY", ""),
        "secret_key": os.getenv("IS3_SECRET_KEY", ""),
        "folder_id": os.getenv("IS3_FOLDER_ID", ""),
        "camera_code": "衷和楼1702",
        "detail_table_code": "camera_result_detail",
        "simple_table_code": "camera_result_simple",
        "timeout": 20
    }
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


class IS3Client:
    """iS3 文件上传和元数据写入客户端"""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.file_base_url = str(config.get("file_base_url", "")).rstrip("/")
        self.project_id = str(config.get("project_id", ""))
        self.access_key = str(config.get("access_key", ""))
        self.secret_key = str(config.get("secret_key", ""))
        self.folder_id = str(config.get("folder_id", ""))
        self.camera_code = str(config.get("camera_code", "衷和楼1702"))
        self.detail_table_code = str(config.get("detail_table_code", "camera_result_detail"))
        self.simple_table_code = str(config.get("simple_table_code", "camera_result_simple"))
        self.timeout = int(config.get("timeout", 20))
        self.api = MetadataAPI(
            base_url=self.base_url,
            prj_id=self.project_id,
            headers={
                "X-Access-Key": self.access_key,
                "X-Secret-Key": self.secret_key,
            },
            folder_id=self.folder_id,
            file_base_url=self.file_base_url,
            timeout=self.timeout,
        )

    def is_available(self) -> bool:
        return all([
            self.enabled,
            self.base_url,
            self.project_id,
            self.access_key,
            self.secret_key,
            self.detail_table_code,
            self.simple_table_code,
        ])

    def upload_file(self, file_path: str) -> Optional[str]:
        if not self.folder_id:
            logger.warning("iS3 folder_id 未配置，跳过文件上传")
            return None

        if not file_path or not os.path.exists(file_path):
            logger.warning(f"待上传文件不存在: {file_path}")
            return None

        try:
            result = self.api.upload_file(file_path)
            if isinstance(result, dict) and result.get("success", False):
                file_data = result.get("data") or {}
                rel_url = file_data.get("url")
                if rel_url:
                    return f"{self.file_base_url}{rel_url}"
                logger.warning(f"iS3 上传成功但缺少 data.url: {result}")
                return None

            logger.warning(f"iS3 文件上传失败: {result}")
            return None
        except Exception as e:
            logger.warning(f"iS3 文件上传异常: {e}")
            return None

    def check_folder_access(self) -> bool:
        if not self.folder_id:
            return False
        try:
            result = self.api.get_file_list(self.folder_id, page_num=1, page_size=1)
            if isinstance(result, dict) and result.get("code") == 200:
                logger.info("iS3 目录校验成功")
                return True
            logger.warning(f"iS3 目录校验失败: {result}")
            return False
        except Exception as e:
            logger.warning(f"iS3 目录校验异常: {e}")
            return False

    def add_data(self, meta_table_code: str, row: dict) -> bool:
        try:
            result = self.api.insert_data(meta_table_code, [row])
            if isinstance(result, dict) and result.get("success", False):
                return True
            logger.warning(f"iS3 写入失败: table={meta_table_code}, resp={result}")
            return False
        except Exception as e:
            logger.warning(f"iS3 写入异常: table={meta_table_code}, err={e}")
            return False

    def upload_detection_result(self, latest_result: dict) -> bool:
        timestamp_text = latest_result.get("timestamp") or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detections = latest_result.get("detections") or []
        detection_count = int(latest_result.get("detection_count", len(detections)))

        image_path = latest_result.get("annotated_image") or latest_result.get("cropped_image")
        camera_img_value = self.upload_file(image_path) if image_path else None

        detail_row = {
            "camera_code": self.camera_code,
            "camera_time": timestamp_text,
            "camera_img": camera_img_value,
            "result_json": json.dumps(detections, ensure_ascii=False),
            "result_num": detection_count
        }

        simple_row = {
            "camera_code": self.camera_code,
            "camera_time": timestamp_text,
            "camera_result": detection_count
        }

        detail_ok = self.add_data(self.detail_table_code, detail_row)
        simple_ok = self.add_data(self.simple_table_code, simple_row)

        if not detail_ok:
            logger.warning("iS3 写入失败: camera_result_detail")
        if not simple_ok:
            logger.warning("iS3 写入失败: camera_result_simple")
        return detail_ok and simple_ok


# ========== 模型加载 ==========
def load_groundingdino_model():
    """加载 GroundingDINO 模型"""
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    project_root = Path(__file__).resolve().parents[0]
    config_file = "groundingdino/config/groundingdino_swinb_cogcoor.py"
    checkpoint_path = project_root / "weights" / "groundingdino_swinb_cogcoor.pth"

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


# ========== 配置文件加载 ==========
def load_camera_config(config_path: str) -> dict:
    """
    加载相机检测配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.info(f"配置文件已加载: {config_path}")
        logger.info(f"  相机名称: {config.get('camera_name', 'N/A')}")
        logger.info(f"  检测区域数: {len(config.get('detection_regions', []))}")
        logger.info(f"  标识物点数: {len(config.get('registration', {}).get('landmark_points', []))}")

        return config

    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


# ========== 多区域裁剪 ==========
def crop_multi_regions(image_np: np.ndarray, regions_config: List[Dict],
                        dx: int = 0, dy: int = 0) -> List[Dict]:
    """
    对图像进行多区域裁剪

    Args:
        image_np: 输入图像
        regions_config: 区域配置列表
        dx: X方向偏移量
        dy: Y方向偏移量

    Returns:
        裁剪结果列表 [{"region_id": "...", "crop": np.ndarray, "crop_params": dict}, ...]
    """
    results = []
    h, w = image_np.shape[:2]

    for region_config in regions_config:
        region_id = region_config.get("region_id", "unknown")
        region_name = region_config.get("region_name", "")

        # 获取裁剪参数
        crop_params = region_config.get("crop_params", {})
        crop_width = crop_params.get("width", CONFIG["crop_width"])
        crop_height = crop_params.get("height", CONFIG["crop_height"])
        crop_x = crop_params.get("x", CONFIG["crop_x"])
        crop_y = crop_params.get("y", CONFIG["crop_y"])

        # 应用偏移量
        if region_config.get("offset_correction", {}).get("enabled", True):
            crop_x = int(crop_x + dx)
            crop_y = int(crop_y + dy)

        # 确保裁剪区域在图像范围内
        crop_x = max(0, min(crop_x, w - crop_width))
        crop_y = max(0, min(crop_y, h - crop_height))

        # 裁剪图像
        cropped = image_np[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        results.append({
            "region_id": region_id,
            "region_name": region_name,
            "crop": cropped,
            "crop_params": {
                "width": crop_width,
                "height": crop_height,
                "x": crop_x,
                "y": crop_y
            }
        })

    return results


# ========== 多区域检测 ==========
def detect_multi_regions(model, image_crops: List[Dict], regions_config: List[Dict],
                         device: str) -> Dict[str, List]:
    """
    对多个裁剪区域进行检测

    Args:
        model: GroundingDINO模型
        image_crops: 裁剪结果列表
        regions_config: 区域配置列表
        device: 设备

    Returns:
        检测结果字典 {"region_id": [detections, ...], ...}
    """
    results = {}

    # 创建区域配置映射
    region_config_map = {r.get("region_id"): r for r in regions_config}

    for crop_data in image_crops:
        region_id = crop_data["region_id"]
        cropped_image = crop_data["crop"]

        # 获取区域检测参数
        region_config = region_config_map.get(region_id, {})
        detection_params = region_config.get("detection_params", {})

        caption = detection_params.get("caption", CONFIG["detection_caption"])
        box_threshold = detection_params.get("box_threshold", CONFIG["box_threshold"])
        text_threshold = detection_params.get("text_threshold", CONFIG["text_threshold"])
        max_box_area = detection_params.get("max_box_area", CONFIG["max_box_area"])
        min_box_area = detection_params.get("min_box_area", CONFIG["min_box_area"])

        # 执行检测
        detections = detect_objects(
            model, cropped_image, caption, box_threshold, text_threshold,
            max_box_area, min_box_area
        )

        results[region_id] = detections

    return results


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


# ========== 多区域结果保存 ==========
def save_multi_region_result(frame_np: np.ndarray, image_crops: List[Dict],
                              detection_results: Dict[str, List], output_dir: Path,
                              timestamp: datetime, camera_name: str = "camera",
                              offset_info: Dict = None) -> List[str]:
    """
    保存多区域检测结果

    Args:
        frame_np: 原始帧
        image_crops: 裁剪结果列表
        detection_results: 检测结果字典
        output_dir: 输出目录
        timestamp: 时间戳
        camera_name: 相机名称
        offset_info: 偏移信息

    Returns:
        保存的JSON文件路径列表
    """
    from groundingdino.util.inference import annotate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    # 保存原始帧
    original_path = output_dir / "original" / f"{filename}.jpg"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(original_path), frame_np)

    json_paths = []
    total_detections = 0

    for crop_data in image_crops:
        region_id = crop_data["region_id"]
        region_name = crop_data["region_name"]
        cropped_image = crop_data["crop"]
        crop_params = crop_data["crop_params"]

        # 创建区域子目录
        region_dir = output_dir / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)

        # 保存裁剪图像
        crop_path = region_dir / f"{filename}_crop.jpg"
        cv2.imwrite(str(crop_path), cropped_image)

        # 获取检测结果
        detections = detection_results.get(region_id, [])
        total_detections += len(detections)

        # 创建带标注的图像
        annotated = cropped_image.copy()
        if detections:
            boxes_list = []
            logits_list = []
            phrases_list = []

            h_img, w_img = cropped_image.shape[:2]
            for det in detections:
                cx = det["center_x"]
                cy = det["center_y"]
                bw = det["width"] / w_img
                bh = det["height"] / h_img

                boxes_list.append([cx, cy, bw, bh])
                logits_list.append(det["confidence"])
                phrases_list.append(det["label"])

            if boxes_list:
                boxes_tensor = torch.tensor(boxes_list)
                logits_tensor = torch.tensor(logits_list)

                annotated = annotate(
                    image_source=cropped_image,
                    boxes=boxes_tensor,
                    logits=logits_tensor,
                    phrases=phrases_list
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            else:
                annotated_rgb = cropped_image
        else:
            annotated_rgb = cropped_image

        # 保存带标注的图像
        annotated_path = region_dir / f"{filename}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_rgb)

        # 保存区域检测结果JSON
        region_result = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "timestamp_unix": timestamp.timestamp(),
            "camera_name": camera_name,
            "region_id": region_id,
            "region_name": region_name,
            "original_image": str(original_path),
            "cropped_image": str(crop_path),
            "annotated_image": str(annotated_path),
            "crop_params": crop_params,
            "offset_correction": offset_info,
            "detection_count": len(detections),
            "detections": detections
        }

        region_json_path = region_dir / f"{filename}.json"
        with open(region_json_path, 'w', encoding='utf-8') as f:
            json.dump(region_result, f, ensure_ascii=False, indent=2)

        json_paths.append(str(region_json_path))

    # 保存汇总结果
    summary_result = {
        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "timestamp_unix": timestamp.timestamp(),
        "camera_name": camera_name,
        "original_image": str(original_path),
        "total_detection_count": total_detections,
        "regions": [
            {
                "region_id": crop_data["region_id"],
                "region_name": crop_data["region_name"],
                "detection_count": len(detection_results.get(crop_data["region_id"], [])),
                "crop_params": crop_data["crop_params"]
            }
            for crop_data in image_crops
        ],
        "offset_correction": offset_info
    }

    summary_json_path = output_dir / "json" / f"{filename}_summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, ensure_ascii=False, indent=2)

    json_paths.append(str(summary_json_path))

    return json_paths


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

    is3_client = IS3Client(CONFIG.get("is3", {}))
    if is3_client.is_available():
        logger.info("iS3 上报已启用")
        is3_client.check_folder_access()
    else:
        logger.warning("iS3 上报未完整配置，将跳过上传")
    logger.info("RTSP 视频流检测服务")
    logger.info("=" * 60)

    # 打印配置
    logger.info("配置参数:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 60)

    # 加载相机配置（如果指定）
    camera_config_path = os.getenv("CAMERA_CONFIG_PATH", "")
    camera_config = {}
    multi_region_mode = False

    if camera_config_path and os.path.exists(camera_config_path):
        camera_config = load_camera_config(camera_config_path)
        multi_region_mode = bool(camera_config.get("detection_regions"))

        if multi_region_mode:
            logger.info("✓ 多区域检测模式已启用")

            # 更新相机配置
            if camera_config.get("camera_name"):
                CONFIG["is3"]["camera_code"] = camera_config["camera_name"]
            if camera_config.get("rtsp_settings", {}).get("url"):
                CONFIG["rtsp_url"] = camera_config["rtsp_settings"]["url"]
            if camera_config.get("rtsp_settings", {}).get("sample_interval"):
                CONFIG["sample_interval"] = camera_config["rtsp_settings"]["sample_interval"]
            if camera_config.get("rtsp_settings", {}).get("output_dir"):
                CONFIG["output_dir"] = camera_config["rtsp_settings"]["output_dir"]
        else:
            logger.warning("配置文件未定义检测区域，使用单区域模式")
    else:
        logger.info("未指定配置文件或文件不存在，使用单区域模式")

    # 初始化图像配准器
    image_registration = None
    current_offset = {"dx": 0, "dy": 0}

    if multi_region_mode and IMAGE_REGISTRATION_AVAILABLE:
        registration_config = camera_config.get("registration", {})
        if registration_config.get("enabled", True):
            reg_method = registration_config.get("method", "DINO_TREE_GRID")
            ref_image_path = registration_config.get("reference_image_path", "")

            if reg_method == "DINO_TREE_GRID":
                # DINO 树木网格配准
                if not SCIPY_AVAILABLE or DinoTreeGridRegistration is None:
                    logger.warning("scipy未安装或DinoTreeGridRegistration不可用，无法使用 DINO 树木网格配准")
                elif ref_image_path and os.path.exists(ref_image_path):
                    reg_params = {
                        "caption": registration_config.get("caption", "tree"),
                        "box_threshold": registration_config.get("box_threshold", 0.3),
                        "text_threshold": registration_config.get("text_threshold", 0.25),
                        "device": CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                        "smoothing_window": registration_config.get("smoothing_window", 5)
                    }

                    # 需要模型，稍后在加载模型后初始化
                    dino_reg_params = reg_params
                    logger.info(f"将使用 DINO 树木网格配准")
                else:
                    logger.warning("未找到参考图像，偏移补偿将被禁用")
            else:
                logger.warning(f"不支持的配准方法: {reg_method}，仅支持 DINO_TREE_GRID")

    # 加载模型
    try:
        model = load_groundingdino_model()
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 初始化 DINO 树木网格配准器（需要模型）
    image_registration = None
    if multi_region_mode and IMAGE_REGISTRATION_AVAILABLE and 'dino_reg_params' in locals():
        registration_config = camera_config.get("registration", {})
        ref_image_path = registration_config.get("reference_image_path", "")

        if ref_image_path and os.path.exists(ref_image_path):
            try:
                image_registration = DinoTreeGridRegistration(model=model, config=dino_reg_params)
                ref_image = cv2.imread(ref_image_path)
                if image_registration.set_reference_image(ref_image):
                    logger.info(f"✓ DINO 树木网格配准器已初始化")
                    logger.info(f"  检测文本: {dino_reg_params['caption']}")
                    logger.info(f"  框阈值: {dino_reg_params['box_threshold']}")
                    logger.info(f"  文本阈值: {dino_reg_params['text_threshold']}")
                else:
                    logger.warning("设置参考图像失败，偏移补偿将被禁用")
                    image_registration = None
            except Exception as e:
                logger.error(f"DINO 树木网格配准器初始化失败: {e}")
                image_registration = None

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
    logger.info(f"  检测模式: {'多区域' if multi_region_mode else '单区域'}")
    if image_registration:
        logger.info(f"  偏移补偿: 启用 ({image_registration.method})")
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
                # 计算偏移量（如果启用）
                dx, dy = 0, 0
                offset_confidence = 0.0

                if image_registration:
                    dx, dy, offset_confidence, offset_debug = image_registration.compute_offset(frame)
                    current_offset = {"dx": dx, "dy": dy, "confidence": offset_confidence}

                    if offset_debug.get("success"):
                        logger.debug(f"  偏移量: dx={dx}, dy={dy}, confidence={offset_confidence:.2f}")
                    else:
                        logger.debug(f"  偏移量计算失败，使用历史值: dx={dx}, dy={dy}")

                if multi_region_mode:
                    # 多区域检测模式
                    detection_regions = camera_config.get("detection_regions", [])

                    # 应用偏移量并进行多区域裁剪
                    image_crops = crop_multi_regions(frame, detection_regions, dx, dy)

                    # 多区域检测
                    detection_results = detect_multi_regions(model, image_crops, detection_regions, CONFIG["device"])

                    # 统计总检测数
                    total_frame_detections = sum(len(detections) for detections in detection_results.values())

                    detection_count += 1
                    total_detected_items += total_frame_detections

                    # 保存多区域结果
                    json_paths = save_multi_region_result(
                        frame, image_crops, detection_results,
                        CONFIG["output_dir"], timestamp,
                        camera_config.get("camera_name", CONFIG["is3"]["camera_code"]),
                        current_offset if image_registration else None
                    )

                    # 输出结果
                    for crop_data in image_crops:
                        region_id = crop_data["region_id"]
                        region_name = crop_data["region_name"]
                        detections = detection_results.get(region_id, [])

                        if detections:
                            logger.info(f"  区域 [{region_name}] 检测到 {len(detections)} 个目标")
                            for i, det in enumerate(detections[:3]):  # 只显示前3个
                                logger.info(f"    [{i+1}] {det['label']} - 置信度: {det['confidence']:.3f}")
                        else:
                            logger.info(f"  区域 [{region_name}] 未检测到目标")

                    logger.info(f"  总检测数: {total_frame_detections}")

                    # 更新汇总（使用第一个区域的JSON作为最新结果）
                    if json_paths:
                        with open(json_paths[-1], 'r', encoding='utf-8') as f:
                            latest_result = json.load(f)
                        update_summary_json(CONFIG["output_dir"], latest_result)

                    # 上传到 iS3 平台（使用汇总结果）
                    if is3_client.is_available():
                        if total_frame_detections < 5:
                            logger.info(f"  iS3 上传跳过：检测到 {total_frame_detections} 个目标（阈值：5）")
                        else:
                            # 构造兼容的上传结果
                            upload_result = {
                                "timestamp": latest_result.get("timestamp"),
                                "detections": [],
                                "detection_count": total_frame_detections,
                                "annotated_image": latest_result.get("original_image")
                            }
                            for region_id, detections in detection_results.items():
                                upload_result["detections"].extend(detections)

                            is3_ok = is3_client.upload_detection_result(upload_result)
                            if is3_ok:
                                logger.info("  iS3 上传成功（文件 + 元数据）")
                            else:
                                logger.warning("  iS3 上传部分或全部失败")

                    logger.info(f"  结果已保存: {len(json_paths)} 个文件")

                else:
                    # 单区域检测模式（原有逻辑）
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

                    # 上传到 iS3 平台
                    if is3_client.is_available():
                        # 检测数量小于5时跳过上传
                        if len(detections) < 5:
                            logger.info(f"  iS3 上传跳过：检测到 {len(detections)} 个目标（阈值：5）")
                        else:
                            is3_ok = is3_client.upload_detection_result(latest_result)
                            if is3_ok:
                                logger.info("  iS3 上传成功（文件 + 元数据）")
                            else:
                                logger.warning("  iS3 上传部分或全部失败")

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
