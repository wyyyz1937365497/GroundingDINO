import argparse
from functools import partial
import cv2
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
import logging

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("警告: ffmpeg-python 未安装，将使用 OpenCV 读取视频")
    print("安装方法: pip install ffmpeg-python")

# 设置日志级别，抑制警告
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow日志（如果有）

import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

# 重定向stderr以抑制FFmpeg警告
class SuppressStdErr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.stderr

import torch

# prepare the environment
# os.system("python setup.py build develop --user")
# os.system("pip install packaging==21.3")
# os.system("pip install gradio==3.50.2")


warnings.filterwarnings("ignore")

import gradio as gr

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download


# Use this command for evaluate the Grounding DINO model
config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    # cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    cache_file = str(Path(__file__).resolve().parents[1] / "weights" / "groundingdino_swint_ogc.pth")
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model = model.to(device)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    print(f"Using inference device: {device}")
    _ = model.eval()
    return model

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae, device=DEVICE)

def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    init_image = input_image.convert("RGB")
    original_size = init_image.size

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    # run grounidng
    boxes, logits, phrases = predict(model, image_tensor, grounding_caption, box_threshold, text_threshold, device=DEVICE)
    annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))


    return image_with_box

def get_video_info(video_path):
    """获取视频信息：帧率、总帧数、时长"""
    try:
        if FFMPEG_AVAILABLE:
            # 使用ffmpeg-python获取视频信息
            with SuppressStdErr():
                probe = ffmpeg.probe(video_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

                fps = eval(video_info['r_frame_rate'])  # 评估分数，例如 "30/1" -> 30.0
                duration = float(probe['format']['duration'])
                total_frames = int(video_info.get('nb_frames', duration * fps))

                return int(fps), total_frames, duration
        else:
            # 回退到OpenCV
            with SuppressStdErr():
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None, None, None

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0

                cap.release()
                return fps, total_frames, duration
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None, None, None

def extract_frames_with_ffmpeg(video_path, output_dir, start_time, end_time, fps, crop_params=None):
    """
    使用ffmpeg从视频中批量提取帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        start_time: 起始时间（秒）
        end_time: 结束时间（秒）
        fps: 原视频帧率
        crop_params: 裁剪参数 (width:height:x:y) 或 None

    Returns:
        提取的帧信息列表 [(frame_idx, image_path, timestamp), ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_infos = []
    try:
        # 使用ffmpeg提取帧到临时目录
        # 使用select='eq(n\,0)'等过滤器来选择特定帧
        temp_pattern = os.path.join(output_dir, "frame_%04d.png")

        cmd = ['ffmpeg', '-i', video_path, '-vf', "select='eq(n\\,0)+eq(n\\,30)+eq(n\\,60)'"]

        # 添加裁剪参数
        if crop_params:
            cmd[3] = f"select='eq(n\\,0)+eq(n\\,30)+eq(n\\,60)',crop={crop_params}"

        cmd.extend(['-vsync', '0', temp_pattern, '-y'])

        # 对于更灵活的抽帧，我们使用Python循环
        # 先获取视频的持续时间
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return {
            'fps': video_fps,
            'total_frames': total_frames,
            'output_dir': output_dir
        }

    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None

def extract_single_frame(video_path, frame_idx, crop_params=None, output_dir=None):
    """
    提取单个帧

    Args:
        video_path: 视频路径
        frame_idx: 帧索引
        crop_params: 裁剪参数
        output_dir: 输出目录

    Returns:
        (frame_idx, image_pil, timestamp)
    """
    try:
        if FFMPEG_AVAILABLE:
            # 使用ffmpeg-python提取帧
            # 首先获取视频的fps来计算时间戳
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps = eval(video_info['r_frame_rate'])

            # 计算时间戳
            timestamp = frame_idx / fps if fps > 0 else frame_idx / 30.0

            # 构建ffmpeg命令提取指定帧
            if crop_params:
                parts = crop_params.split(':')
                w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                process = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output('pipe:', format='image2', vframes=1, vf=f'crop={w}:{h}:{x}:{y}')
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
            else:
                process = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output('pipe:', format='image2', vframes=1)
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

            # 读取输出
            out, err = process.communicate()

            if process.returncode != 0:
                # 如果ffmpeg失败，回退到OpenCV
                raise Exception(f"FFmpeg extraction failed: {err.decode()}")

            # 从字节流读取图像
            image = Image.open(BytesIO(out))

            return (frame_idx, image, timestamp)

        else:
            # 回退到OpenCV
            with SuppressStdErr():
                cap = cv2.VideoCapture(video_path)

                # 设置帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                # 读取帧
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    return None

                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 裁剪
                if crop_params:
                    parts = crop_params.split(':')
                    w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    # 确保裁剪区域在图像范围内
                    img_h, img_w = frame_rgb.shape[:2]
                    x = min(x, img_w - w)
                    y = min(y, img_h - h)
                    frame_rgb = frame_rgb[y:y+h, x:x+w]

                # 获取视频fps来计算准确的时间戳
                cap_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                timestamp = frame_idx / cap_fps if cap_fps > 0 else frame_idx / 30.0

                return (frame_idx, Image.fromarray(frame_rgb), timestamp)

    except Exception as e:
        print(f"Error extracting frame {frame_idx}: {e}")
        return None

def process_frame_detection(frame_info, grounding_caption, box_threshold, text_threshold,
                            max_box_area=None, min_box_area=None):
    """
    处理单个帧的检测（用于多线程）

    Args:
        frame_info: (frame_idx, image_pil, timestamp)
        grounding_caption: 检测提示词
        box_threshold: 边界框阈值
        text_threshold: 文本阈值
        max_box_area: 最大框面积（像素），超过此面积的框将被过滤（用于过滤大物体如树木）
        min_box_area: 最小框面积（像素），小于此面积的框将被过滤（用于过滤噪声）

    Returns:
        检测结果字典
    """
    frame_idx, image, timestamp = frame_info

    try:
        # 转换为RGB
        image_rgb = image.convert("RGB")

        # 图像变换
        _, image_tensor = image_transform_grounding(image_rgb)

        # 运行检测
        boxes, logits, phrases = predict(
            model,
            image_tensor,
            grounding_caption,
            box_threshold,
            text_threshold,
            device=DEVICE
        )

        # 按框大小筛选
        filtered_boxes = []
        filtered_logits = []
        filtered_phrases = []
        filtered_areas = []
        # 保存框的详细信息（位置、尺寸）
        filtered_box_details = []

        if boxes is not None and len(boxes) > 0:
            img_h, img_w = image_rgb.size[1], image_rgb.size[0]

            for box, logit, phrase in zip(boxes, logits, phrases):
                # box格式: [cx, cy, w, h] (归一化坐标)
                cx, cy, bw, bh = box

                # 计算实际像素尺寸
                box_w = bw * img_w
                box_h = bh * img_h
                box_area = box_w * box_h

                # 计算实际像素坐标（左上角和右下角）
                x1 = cx * img_w - box_w / 2
                y1 = cy * img_h - box_h / 2
                x2 = cx * img_w + box_w / 2
                y2 = cy * img_h + box_h / 2

                # 筛选条件
                if max_box_area is not None and box_area > max_box_area:
                    continue  # 跳过过大的框（如树木）
                if min_box_area is not None and box_area < min_box_area:
                    continue  # 跳过过小的框（如噪声）

                filtered_boxes.append(box)
                filtered_logits.append(logit)
                filtered_phrases.append(phrase)
                filtered_areas.append(box_area)

                # 保存框的详细信息
                filtered_box_details.append({
                    # 归一化坐标 (0-1)
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "width_norm": float(bw),
                    "height_norm": float(bh),
                    # 实际像素坐标
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    # 实际像素尺寸
                    "width": float(box_w),
                    "height": float(box_h),
                    "area": float(box_area)
                })

            # 转换为tensor
            if len(filtered_boxes) > 0:
                filtered_boxes = torch.stack(filtered_boxes)
                filtered_logits = torch.stack(filtered_logits)
            else:
                filtered_boxes = None
                filtered_logits = None
        else:
            filtered_boxes = boxes
            filtered_logits = logits

        # 统计数量（使用筛选后的）
        count = len(filtered_boxes) if filtered_boxes is not None else 0

        # 创建标注图像（使用筛选后的框）
        image_pil = image_transform_grounding_for_vis(image_rgb)
        annotated_frame = annotate(
            image_source=np.asarray(image_pil),
            boxes=filtered_boxes,
            logits=filtered_logits,
            phrases=filtered_phrases
        )
        image_with_box = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        return {
            "frame": frame_idx,
            "timestamp": f"{timestamp:.2f}s",
            "count": count,
            "detected_items": [
                {
                    "label": phrase,
                    "confidence": float(logit),
                    **box_details
                }
                for phrase, logit, box_details in zip(filtered_phrases, filtered_logits, filtered_box_details)
            ],
            "annotated_image": Image.fromarray(image_with_box)
        }

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return {
            "frame": frame_idx,
            "timestamp": f"{timestamp:.2f}s",
            "count": 0,
            "detected_items": [],
            "error": str(e),
            "annotated_image": image
        }

# ========== 第一阶段：抽帧 ==========

def extract_frames_stage(
    video_upload,
    video_dropdown,
    start_time_str,
    frame_interval,
    end_time_str,
    crop_width,
    crop_height,
    crop_x,
    crop_y,
    enable_crop,
    progress=gr.Progress()
):
    """
    第一阶段：从视频中提取帧（固定间隔抽帧）

    Returns:
        (stage_data_json, preview_images, info_text)
    """
    # 确定使用哪个视频源
    video_file = video_dropdown if video_dropdown else video_upload

    if video_file is None:
        return "{}", [], "请上传视频或从列表中选择"

    try:
        # 获取视频文件路径
        if isinstance(video_file, str):
            video_path = video_file
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        else:
            video_path = str(video_file)

        # 检查文件是否存在
        if not os.path.exists(video_path):
            return "{}", [], f"视频文件不存在: {video_path}"

        # 获取视频信息
        fps, total_frames, duration = get_video_info(video_path)
        if fps is None:
            return "{}", [], "无法读取视频信息"

        # 解析时间参数
        try:
            start_time = parse_time_string(start_time_str)
            end_time = parse_time_string(end_time_str) if end_time_str else duration
        except ValueError as e:
            return f"{{}}", [], f"时间格式错误: {e}"

        # 转换为帧数
        start_frame = int(start_time * fps)
        end_frame = int(min(end_time, duration) * fps)

        if start_frame >= total_frames:
            return "{}", [], f"起始帧超出视频总帧数 (视频共{total_frames}帧)"

        end_frame = min(end_frame, total_frames)

        # 构建裁剪参数
        crop_params = None
        if enable_crop:
            crop_params = f"{crop_width}:{crop_height}:{crop_x}:{crop_y}"

        # 创建临时目录保存帧
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")

        # 固定间隔抽帧
        frame_indices = list(range(start_frame, end_frame, frame_interval))
        total_to_extract = len(frame_indices)

        extracted_frames = []
        preview_images = []

        for i, frame_idx in enumerate(frame_indices):
            timestamp = frame_idx / fps
            result = extract_single_frame(video_path, frame_idx, crop_params, temp_dir)

            if result:
                frame_idx, image, timestamp = result
                extracted_frames.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "temp_dir": temp_dir
                })

                # 保存前5帧作为预览
                if len(preview_images) < 5:
                    preview_images.append(image)

            # 更新进度
            progress((i + 1) / total_to_extract, desc=f"提取帧 {i+1}/{total_to_extract}")

        # 准备阶段数据（不需要保存实际图像，只保存索引）
        stage_data = {
            "video_path": video_path,
            "temp_dir": temp_dir,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_interval": frame_interval,
            "crop_params": crop_params,
            "frame_indices": frame_indices,
            "total_to_process": len(frame_indices)
        }

        info_text = (
            f"✓ 抽帧完成! (固定间隔: 每{frame_interval}帧)\n"
            f"视频信息: {total_frames}帧, {fps}fps, {duration:.1f}秒\n"
            f"处理范围: 帧 {start_frame} - {end_frame}\n"
            f"已提取帧数: {len(extracted_frames)}\n"
            f"临时目录: {temp_dir}\n"
            f"\n请点击'开始检测'进行物品检测"
        )

        return json.dumps(stage_data), preview_images, info_text

    except Exception as e:
        import traceback
        error_msg = f"抽帧出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return "{}", [], error_msg

# ========== 第二阶段：检测 ==========

def detect_items_stage(
    stage_data_json,
    grounding_caption,
    box_threshold,
    text_threshold,
    num_workers,
    max_box_area,
    min_box_area,
    progress=gr.Progress()
):
    """
    第二阶段：对提取的帧进行物品检测（多线程并行）

    Args:
        stage_data_json: 第一阶段返回的数据
        grounding_caption: 检测提示词
        box_threshold: 边界框阈值
        text_threshold: 文本阈值
        num_workers: 并行工作线程数
        max_box_area: 最大框面积（像素），超过此面积的框将被过滤
        min_box_area: 最小框面积（像素），小于此面积的框将被过滤

    Returns:
        (json_output, sample_images, info_text)
    """
    if not stage_data_json or stage_data_json == "{}":
        return "{}", [], "请先完成抽帧阶段"

    if not grounding_caption or grounding_caption.strip() == "":
        return "{}", [], "请输入检测提示词"

    try:
        # 解析阶段数据
        stage_data = json.loads(stage_data_json)

        video_path = stage_data["video_path"]
        fps = stage_data["fps"]
        frame_indices = stage_data["frame_indices"]
        crop_params = stage_data.get("crop_params")

        # 准备检测结果
        results = []
        sample_images = []

        # 重新提取帧并检测（多线程）
        total_to_process = len(frame_indices)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_frame = {}
            for i, frame_idx in enumerate(frame_indices):
                # 提取帧
                frame_info = extract_single_frame(video_path, frame_idx, crop_params)
                if frame_info:
                    future = executor.submit(
                        process_frame_detection,
                        frame_info,
                        grounding_caption,
                        box_threshold,
                        text_threshold,
                        max_box_area,
                        min_box_area
                    )
                    future_to_frame[future] = frame_idx

                # 每10个任务更新一次进度（避免过于频繁）
                if (i + 1) % 10 == 0:
                    progress((i + 1) / total_to_process, desc=f"提交检测任务 {i+1}/{total_to_process}")

            # 收集结果
            completed = 0
            for future in as_completed(future_to_frame):
                result = future.result()
                if result:
                    # 保留完整的结果，包括标注图片
                    results.append(result)

                    # 保存前5帧作为样本显示
                    if "annotated_image" in result and len(sample_images) < 5:
                        sample_images.append(result["annotated_image"])

                completed += 1
                # 确保进度不超过100%
                progress_val = min(completed / total_to_process, 1.0)
                progress(progress_val, desc=f"检测完成 {completed}/{total_to_process}")

        # 按帧号排序
        results.sort(key=lambda x: x["frame"])

        # 获取视频名称和时间戳
        video_name = Path(video_path).stem
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建输出目录
        output_dir = Path(__file__).resolve().parents[1] / "output" / "detection_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建标注图片目录（每次处理前清空）
        annotated_images_dir = output_dir / "latest_annotated_frames"
        # 清空目录中的旧文件
        if annotated_images_dir.exists():
            import shutil
            shutil.rmtree(annotated_images_dir)
        annotated_images_dir.mkdir(parents=True, exist_ok=True)

        # 保存所有标注后的图片
        annotated_images_paths = []
        for result in results:
            if "annotated_image" in result:
                frame_idx = result["frame"]
                # 使用固定文件名模式，方便覆盖
                image_filename = f"frame_{frame_idx:06d}.png"
                image_filepath = annotated_images_dir / image_filename

                # 保存图片
                try:
                    result["annotated_image"].save(image_filepath)
                    annotated_images_paths.append(str(image_filepath))
                except Exception as e:
                    print(f"保存图片 {image_filename} 失败: {e}")

        # 创建ZIP文件包含所有标注图片
        zip_filepath = None
        if annotated_images_paths:
            try:
                import zipfile
                zip_filename = f"{video_name}_annotated_frames_{timestamp_str}.zip"
                zip_filepath = output_dir / zip_filename

                with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for image_path in annotated_images_paths:
                        arcname = Path(image_path).name
                        zipf.write(image_path, arcname)

                print(f"✓ ZIP文件已创建: {zip_filepath}")
            except Exception as e:
                print(f"创建ZIP文件失败: {e}")
                zip_filepath = None

        # 移除annotated_image字段后准备JSON数据
        results_for_json = []
        for result in results:
            result_copy = result.copy()
            if "annotated_image" in result_copy:
                del result_copy["annotated_image"]
            if "error" in result_copy:
                del result_copy["error"]
            results_for_json.append(result_copy)

        # 生成JSON文件名
        json_filename = f"{video_name}_{timestamp_str}.json"
        json_filepath = output_dir / json_filename

        # 准备输出数据
        output_data = {
            "video_name": video_name,
            "video_path": video_path,
            "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotated_images_dir": str(annotated_images_dir),
            "annotated_images_count": len(annotated_images_paths),
            "video_info": {
                "fps": fps,
                "total_frames": stage_data["total_frames"],
                "duration": f"{stage_data['duration']:.2f}s",
                "start_frame": stage_data["start_frame"],
                "end_frame": stage_data["end_frame"]
            },
            "detection_params": {
                "caption": grounding_caption,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "frame_interval": stage_data["frame_interval"],
                "crop_enabled": crop_params is not None,
                "crop_params": crop_params,
                "num_workers": num_workers
            },
            "statistics": {
                "total_frames_processed": len(results),
                "total_items_detected": sum(r["count"] for r in results),
                "average_items_per_frame": sum(r["count"] for r in results) / len(results) if results else 0
            },
            "results": results_for_json
        }

        # 转换为JSON
        json_output = json.dumps(output_data, ensure_ascii=False, indent=2)

        # 保存JSON文件
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"✓ JSON结果已保存到: {json_filepath}")
        except Exception as e:
            print(f"保存JSON文件失败: {e}")

        # 统计信息
        total_detected = output_data["statistics"]["total_items_detected"]
        avg_count = output_data["statistics"]["average_items_per_frame"]

        zip_info = f"\n✓ ZIP文件已保存:\n{zip_filepath}" if zip_filepath else ""
        info_text = (
            f"✓ 检测完成!\n"
            f"视频信息: {stage_data['total_frames']}帧, {fps}fps, {stage_data['duration']:.1f}秒\n"
            f"处理范围: 帧 {stage_data['start_frame']} - {stage_data['end_frame']}\n"
            f"抽帧间隔: 每{stage_data['frame_interval']}帧\n"
            f"处理帧数: {len(results)}\n"
            f"并行线程: {num_workers}\n"
            f"总检测数量: {total_detected}\n"
            f"平均每帧: {avg_count:.1f}个\n"
            f"\n✓ JSON已保存:\n{json_filepath}\n"
            f"✓ 标注图片已保存:\n{annotated_images_dir}\n"
            f"共 {len(annotated_images_paths)} 张图片"
            f"{zip_info}"
        )

        return json_output, sample_images, info_text, str(json_filepath), str(zip_filepath) if zip_filepath else None

    except Exception as e:
        import traceback
        error_msg = f"检测出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return "{}", [], error_msg

def parse_time_string(time_str):
    """
    解析时间字符串，支持格式：
    - "HH:MM:SS"
    - "MM:SS"
    - 秒数（整数或浮点数）
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)

    time_str = time_str.strip()

    # 尝试直接解析为数字
    try:
        return float(time_str)
    except ValueError:
        pass

    # 解析 "HH:MM:SS" 或 "MM:SS" 格式
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"无法解析时间格式: {time_str}")

def list_input_videos():
    """列出input_video目录中的视频文件"""
    input_dir = Path(__file__).resolve().parents[1] / "input_video"
    if input_dir.exists():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        videos = [
            str(f) for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        return videos
    return []

def update_video_choices():
    """更新视频选择下拉框"""
    videos = list_input_videos()
    return gr.update(choices=videos if videos else [], value=videos[0] if videos else None)

# 保存阶段数据
stage_data_state = gr.State(None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)")
        gr.Markdown("### Open-World Detection with Grounding DINO")

        with gr.Tabs():
            # 图像检测标签页
            with gr.Tab("图像检测"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(source='upload', type="pil")
                        grounding_caption = gr.Textbox(label="Detection Prompt", value="object")
                        run_button = gr.Button(label="Run")
                        with gr.Accordion("Advanced options", open=False):
                            box_threshold = gr.Slider(
                                label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                            )
                            text_threshold = gr.Slider(
                                label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                            )

                    with gr.Column():
                        image_gallery = gr.outputs.Image(
                            type="pil",
                        ).style(full_width=True, full_height=True)

                run_button.click(fn=run_grounding, inputs=[
                                input_image, grounding_caption, box_threshold, text_threshold], outputs=[image_gallery])

            # 视频检测标签页 - 分为两个阶段
            with gr.Tab("视频检测（两阶段）"):
                if not FFMPEG_AVAILABLE:
                    gr.Warning("⚠️ ffmpeg-python 未安装，将使用 OpenCV 读取视频。某些视频格式可能不兼容。")
                    gr.Markdown("**安装命令**: `pip install ffmpeg-python`")

                gr.Markdown("### 分阶段处理：1.抽帧 → 2.检测（支持多线程并行）")

                with gr.Row():
                    with gr.Column():
                        # 视频输入
                        video_upload = gr.File(
                            label="方式1: 上传视频",
                            file_types=["video"]
                        )
                        with gr.Row():
                            video_dropdown = gr.Dropdown(
                                label="方式2: 选择input_video目录中的视频",
                                choices=[],
                                interactive=True
                            )
                            refresh_videos_btn = gr.Button("刷新列表", size="sm")

                        gr.Markdown("---")

                        # 阶段1：抽帧参数
                        gr.Markdown("### 第一阶段：抽帧参数")

                        with gr.Accordion("视频处理参数", open=True):
                            start_time = gr.Textbox(
                                label="起始时间",
                                value="00:00:00",
                                placeholder="格式: HH:MM:SS 或 MM:SS 或 秒数"
                            )
                            end_time = gr.Textbox(
                                label="结束时间 (可选)",
                                value="",
                                placeholder="格式: HH:MM:SS 或 MM:SS 或 秒数，留空则处理到视频结尾"
                            )
                            frame_interval = gr.Slider(
                                label="抽帧间隔 (每N帧抽取一帧)",
                                minimum=1,
                                maximum=300,
                                value=30,
                                step=1
                            )

                        with gr.Accordion("裁剪参数", open=True):
                            enable_crop = gr.Checkbox(label="启用裁剪", value=True)
                            crop_width = gr.Slider(
                                label="裁剪宽度",
                                minimum=100,
                                maximum=1920,
                                value=550,
                                step=10
                            )
                            crop_height = gr.Slider(
                                label="裁剪高度",
                                minimum=100,
                                maximum=1920,
                                value=870,
                                step=10
                            )
                            crop_x = gr.Slider(
                                label="裁剪起始X",
                                minimum=0,
                                maximum=1920,
                                value=910,
                                step=10
                            )
                            crop_y = gr.Slider(
                                label="裁剪起始Y",
                                minimum=0,
                                maximum=1920,
                                value=530,
                                step=10
                            )

                        # 阶段1按钮
                        extract_frames_btn = gr.Button("🎬 第一阶段：开始抽帧", variant="primary", size="lg")

                        gr.Markdown("---")

                        # 阶段2：检测参数
                        gr.Markdown("### 第二阶段：检测参数")
                        grounding_caption_video = gr.Textbox(
                            label="Detection Prompt",
                            value="bike",
                            placeholder="例如: person, car, bottle"
                        )

                        with gr.Accordion("检测阈值", open=False):
                            box_threshold_video = gr.Slider(
                                label="Box Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.12,
                                step=0.01
                            )
                            text_threshold_video = gr.Slider(
                                label="Text Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                value=1,
                                step=0.01
                            )

                        num_workers = gr.Slider(
                            label="并行线程数",
                            minimum=1,
                            maximum=8,
                            value=8,
                            step=1,
                            info="建议根据CPU核心数设置"
                        )

                        with gr.Accordion("框大小筛选（按面积过滤）", open=True):
                            max_box_area = gr.Slider(
                                label="最大框面积（像素²）",
                                minimum=0,
                                maximum=500000,
                                value=6000,
                                step=5000,
                                info="超过此面积的框将被过滤（用于过滤大物体如树木）。设为0表示不限制。100000≈300x300像素"
                            )
                            min_box_area = gr.Slider(
                                label="最小框面积（像素²）",
                                minimum=0,
                                maximum=10000,
                                value=150,
                                step=100,
                                info="小于此面积的框将被过滤（用于过滤噪声）。设为0表示不限制。500≈22x22像素"
                            )
                            gr.Markdown("""
                            **说明：**
                            - 车辆的识别框通常较小（几百到几万像素）
                            - 树木的识别框通常很大（几十万像素以上）
                            - 通过设置最大面积可以过滤掉大的树木检测框
                            - 通过设置最小面积可以过滤掉小的噪声
                            """)

                        # 阶段2按钮
                        detect_items_btn = gr.Button("🔍 第二阶段：开始检测", variant="primary", size="lg")

                    with gr.Column():
                        # 阶段数据状态（隐藏）
                        stage_data_json = gr.Textbox(
                            label="阶段数据 (内部使用)",
                            visible=False,
                            value="{}"
                        )

                        # 输出结果
                        stage_info_output = gr.Textbox(
                            label="处理信息",
                            lines=8,
                            interactive=False
                        )

                        frame_preview = gr.Gallery(
                            label="抽帧预览 (前5帧)",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="auto"
                        )

                        json_output = gr.Textbox(
                            label="检测结果 (JSON)",
                            lines=10,
                            interactive=False
                        )

                        with gr.Row():
                            json_download = gr.File(
                                label="下载JSON文件",
                                visible=False
                            )
                            zip_download = gr.File(
                                label="下载标注图片ZIP",
                                visible=False
                            )

                        video_gallery = gr.Gallery(
                            label="检测结果样本 (前5帧)",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height="auto"
                        )

                # 绑定事件
                refresh_videos_btn.click(
                    fn=update_video_choices,
                    outputs=[video_dropdown]
                )

                # 第一阶段：抽帧
                extract_frames_btn.click(
                    fn=extract_frames_stage,
                    inputs=[
                        video_upload,
                        video_dropdown,
                        start_time,
                        frame_interval,
                        end_time,
                        crop_width,
                        crop_height,
                        crop_x,
                        crop_y,
                        enable_crop
                    ],
                    outputs=[stage_data_json, frame_preview, stage_info_output]
                )

                # 第二阶段：检测
                detect_items_btn.click(
                    fn=detect_items_stage,
                    inputs=[
                        stage_data_json,
                        grounding_caption_video,
                        box_threshold_video,
                        text_threshold_video,
                        num_workers,
                        max_box_area,
                        min_box_area
                    ],
                    outputs=[json_output, video_gallery, stage_info_output, json_download, zip_download]
                )


    block.launch(server_name='0.0.0.0', server_port=19555, debug=args.debug, share=False)
