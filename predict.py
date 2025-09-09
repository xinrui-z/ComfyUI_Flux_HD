import os
import shutil
import json
import mimetypes
import logging
from PIL import Image
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from weights_downloader import WeightsDownloader
from config import config
import requests
import tempfile
from urllib.parse import urlparse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局配置（适配Flux工作流）
os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/webm", ".webm")

# 目录配置（固定临时路径，避免权限问题）
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# 图像格式限制（与Flux工作流兼容）
SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
MAX_IMAGE_SIZE = (1024, 1024)  # 适配Flux模型输入尺寸

# 加载Flux工作流模板（请确保该JSON文件路径正确）
with open("examples/api_workflows/flux_enlargement_hd_api.json", "r", encoding="utf-8") as file:
    FIXED_FLUX_WORKFLOW = file.read()


class Predictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.DEFAULT_IMAGE_PATH = os.path.join(self.PROJECT_ROOT, "ComfyUI", "input", "input.png")
        self.comfyUI = None
        self.workflow_template = FIXED_FLUX_WORKFLOW  # 加载Flux工作流
        
        # 核心状态变量
        self.base_workflow = None  # 解析后的工作流对象
        self.default_prompt = None  # 工作流默认Prompt
        self.comfyui_connected = False  # ComfyUI连接状态
        self.fixed_input_path = os.path.join(INPUT_DIR, "input.png")  # 固定输入图片路径
        self.default_image_ready = False  # 默认图片就绪状态

    def setup(self, weights: Optional[str] = None):
        """初始化：目录创建、默认图片准备、权重下载、ComfyUI启动"""
        # 1. 并行执行本地初始化和权重下载（提升效率）
        def local_init_task():
            # 创建必要目录
            for directory in ALL_DIRECTORIES:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"[Setup] 初始化目录: {directory}")
            
            # 准备默认图片 + 更新工作流输入路径
            self._prepare_default_image()
            if not self.default_image_ready:
                raise RuntimeError("[Setup] 默认图片准备失败，无法启动服务")
            logger.info(f"[Setup] 默认图片就绪: {self.fixed_input_path}")
            return True

        # 线程池并行处理
        with ThreadPoolExecutor(max_workers=2) as executor:
            local_future = executor.submit(local_init_task)
            weight_future = executor.submit(self._handle_user_weights, weights) if weights else None

            # 等待任务完成
            local_future.result()
            if weight_future:
                weight_future.result()

        # 2. 启动ComfyUI服务并加载工作流
        self.comfyUI = ComfyUI("127.0.0.1:8188")  # ComfyUI本地地址
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)  # 绑定输入输出目录
        
        # 解析并验证工作流
        self.base_workflow = self.comfyUI.load_workflow(self.workflow_template)
        self._validate_workflow()
        
        # 提取默认Prompt（从Flux工作流的CR Text节点）
        self.default_prompt = self._extract_default_prompt()
        logger.info(f"[Setup] 提取默认Prompt（前50字符）: {self.default_prompt[:50]}...")

        # 3. 建立ComfyUI连接
        self.comfyUI.connect()
        self.comfyui_connected = True
        logger.info(f"[Setup] 服务初始化完成，可接收请求")

    def _prepare_default_image(self):
        """准备默认图片，并更新工作流中LoadImage节点（ID=233）的路径"""
        try:
            # 优先使用项目自带的默认图片
            if os.path.exists(self.DEFAULT_IMAGE_PATH):
                shutil.copy(self.DEFAULT_IMAGE_PATH, self.fixed_input_path)
                logger.info(f"[Setup] 复制项目默认图片到: {self.fixed_input_path}")
            # 无自带图片时，创建1x1白色占位图
            else:
                with Image.new('RGB', (1, 1), color='white') as img:
                    img.save(self.fixed_input_path)
                logger.info(f"[Setup] 创建占位默认图片到: {self.fixed_input_path}")
            
            # 关键：更新工作流中ID=233（LoadImage）的输入路径
            workflow_dict = json.loads(self.workflow_template)
            if "233" in workflow_dict:
                workflow_dict["233"]["inputs"]["image"] = self.fixed_input_path
                self.workflow_template = json.dumps(workflow_dict, ensure_ascii=False)
                logger.info(f"[Setup] 工作流输入节点（ID=233）路径已更新")
            else:
                raise ValueError("[Setup] 工作流缺失LoadImage节点（ID=233）")
            
            self.default_image_ready = True
        except Exception as e:
            logger.error(f"[Setup] 准备默认图片失败: {str(e)}")
            self.default_image_ready = False

    def _handle_user_weights(self, weights: str):
        """下载并部署用户自定义权重（适配Flux模型目录结构）"""
        try:
            # 解析权重URL（支持Replicate交付链接）
            weights_url = weights.url if (hasattr(weights, "url") and weights.url.startswith("http")) else \
                f"https://replicate.delivery/{weights.url}" if hasattr(weights, "url") else weights
            
            logger.info(f"[Setup] 开始下载权重: {weights_url}")
            WeightsDownloader.download("weights.tar", weights_url, config["USER_WEIGHTS_PATH"])

            # 权重目录映射（Flux模型需按类型分类存放）
            weight_mapping = {
                "unet": os.path.join(config["MODELS_PATH"], "unet"),
                "clip": os.path.join(config["MODELS_PATH"], "clip"),
                "vae": os.path.join(config["MODELS_PATH"], "vae"),
                "upscale": os.path.join(config["MODELS_PATH"], "upscale_models")
            }

            # 移动权重到对应目录（跳过已存在文件）
            for weight_type, dst_dir in weight_mapping.items():
                src_dir = os.path.join(config["USER_WEIGHTS_PATH"], weight_type)
                if os.path.exists(src_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for file_name in os.listdir(src_dir):
                        src_path = os.path.join(src_dir, file_name)
                        dst_path = os.path.join(dst_dir, file_name)
                        if not os.path.exists(dst_path):
                            shutil.move(src_path, dst_path)
                            logger.info(f"[Setup] 移动{weight_type}权重: {file_name}")
                else:
                    logger.warning(f"[Setup] 未找到{weight_type}权重目录: {src_dir}")
        except Exception as e:
            logger.error(f"[Setup] 权重处理失败: {str(e)}")
            raise

    def _validate_workflow(self):
        """验证Flux工作流的核心节点是否齐全"""
        required_nodes = [
            {"id": "233", "type": "LoadImage"},          # 图片输入节点
            {"id": "249", "type": "CR Text"},            # Prompt输入节点
            {"id": "42", "type": "CLIPTextEncode"},      # 文本编码节点
            {"id": "77", "type": "UNETLoader"},          # UNet模型加载节点
            {"id": "31", "type": "VAELoader"},           # VAE模型加载节点
            {"id": "128", "type": "SaveImage"}           # 图片输出节点
        ]

        missing = []
        for node in required_nodes:
            workflow_node = self.base_workflow.get(node["id"])
            if not workflow_node or workflow_node.get("class_type") != node["type"]:
                missing.append(f"ID:{node['id']}（{node['type']}）")
        
        if missing:
            raise ValueError(f"[Setup] 工作流无效，缺失核心节点: {', '.join(missing)}")

    def _extract_default_prompt(self) -> str:
        """从Flux工作流的CR Text节点（ID=249）提取默认Prompt"""
        for node_id, node_data in self.base_workflow.items():
            if node_id == "249" and node_data.get("class_type") == "CR Text" and "text" in node_data["inputs"]:
                return node_data["inputs"]["text"].strip()
        raise ValueError("[Setup] 无法提取默认Prompt，工作流缺失CR Text节点（ID=249）")

    def _download_image_from_url(self, image_url: str) -> Optional[str]:
        """从URL下载图片到临时文件"""
        try:
            # 验证URL格式
            if not urlparse(image_url).scheme.startswith("http"):
                raise ValueError(f"无效URL格式: {image_url}")
            
            # 下载图片（流处理，避免内存占用过高）
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            with requests.get(image_url, headers=headers, timeout=30, stream=True) as resp:
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    temp_path = tmp.name
            
            # 验证图片有效性
            with Image.open(temp_path) as img:
                img.verify()
            logger.info(f"[Predict] URL图片下载成功: {image_url}")
            return temp_path
        except Exception as e:
            logger.error(f"[Predict] URL图片下载失败: {str(e)}")
            if "temp_path" in locals():
                os.unlink(temp_path)
            return None

    def _get_file_extension(self, file_path: str) -> Optional[str]:
        """多维度验证图片格式（后缀→PIL→文件签名）"""
        # 1. 优先检查文件后缀
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_IMAGE_TYPES:
            return ext
        
        # 2. 用PIL识别图片格式
        try:
            with Image.open(file_path) as img:
                fmt = img.format.lower()
                if fmt in ["jpeg", "jpg", "png", "webp"]:
                    return f".{fmt}"
        except Exception as e:
            logger.warning(f"[Predict] PIL识别格式失败: {str(e)}")
        
        # 3. 用文件签名识别（兜底）
        try:
            with open(file_path, "rb") as f:
                sig = f.read(12)
            if sig.startswith(b"\xff\xd8\xff"):
                return ".jpg"
            elif sig.startswith(b"\x89PNG\r\n\x1a\n"):
                return ".png"
            elif sig.startswith(b"RIFF") and sig[8:12] == b"WEBP":
                return ".webp"
        except Exception as e:
            logger.warning(f"[Predict] 签名识别格式失败: {str(e)}")
        
        logger.warning(f"[Predict] 不支持的文件格式: {file_path}")
        return None

    def _handle_input_file(self, input_file: Optional[Union[Path, str]]) -> bool:
        """处理用户输入图片（URL/本地文件），覆盖到固定输入路径"""
        if not input_file:
            logger.info(f"[Predict] 未提供输入图片，使用默认图片")
            return True
        
        source_path = None
        try:
            # 处理URL输入
            if isinstance(input_file, str) and input_file.startswith(("http://", "https://")):
                source_path = self._download_image_from_url(input_file)
                if not source_path:
                    raise ValueError(f"URL图片处理失败: {input_file}")
            # 处理本地文件输入
            else:
                source_path = str(input_file)
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"本地文件不存在: {source_path}")
            
            # 验证图片格式
            file_ext = self._get_file_extension(source_path)
            if not file_ext or file_ext not in SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"不支持的图片格式: {file_ext}（仅支持{SUPPORTED_IMAGE_TYPES}）")
            
            # 压缩图片到最大尺寸（适配Flux模型）
            with Image.open(source_path) as img:
                img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                img.save(self.fixed_input_path)
            
            logger.info(f"[Predict] 输入图片已生效: {source_path} → {self.fixed_input_path}")
            return True
        except Exception as e:
            logger.error(f"[Predict] 输入图片处理失败: {str(e)}")
            # 清理临时文件（仅URL下载的图片）
            if source_path and isinstance(input_file, str) and input_file.startswith(("http://", "https://")):
                try:
                    os.unlink(source_path)
                except:
                    pass
            return False

    def _update_workflow_prompt(self, prompt: str) -> dict:
        """更新Flux工作流的Prompt（修改CR Text节点ID=249）"""
        updated_workflow = {k: v.copy() for k, v in self.base_workflow.items()}
        for node_id, node_data in updated_workflow.items():
            if node_id == "249" and node_data.get("class_type") == "CR Text":
                node_data["inputs"]["text"] = prompt.strip()
                logger.info(f"[Predict] Prompt已更新（前50字符）: {prompt[:50]}...")
                return updated_workflow
        raise ValueError("[Predict] 无法更新Prompt，工作流缺失CR Text节点（ID=249）")

    def predict(
        self,
        input_file: Optional[str] = Input(
            description="输入图片：支持HTTP/HTTPS URL或本地上传路径（格式：jpg/jpeg/png/webp），未提供则用默认图片",
            default=None
        ),
        prompt: str = Input(
            description="Input Prompt",
            default="high quality, detailed, photograph, hd, 8k, 4k, sharp, highly detailed"
        )
    ) -> List[Path]:
        """核心推理方法：处理输入→更新工作流→执行生成→返回结果"""
        # 前置检查：服务是否就绪
        if not self.comfyui_connected or not self.default_image_ready:
            raise RuntimeError("[Predict] 服务未初始化，请先执行setup()")
        
        # 验证Prompt有效性
        user_prompt = prompt.strip()
        if not user_prompt:
            raise ValueError("[Predict] Prompt不能为空")

        # 并行处理：输入图片 + Prompt更新（提升效率）
        with ThreadPoolExecutor(max_workers=2) as executor:
            input_future = executor.submit(self._handle_input_file, input_file)
            workflow_future = executor.submit(self._update_workflow_prompt, user_prompt)

            # 获取处理结果
            if not input_future.result():
                raise ValueError("[Predict] 输入图片处理失败，无法继续")
            current_workflow = workflow_future.result()

        # 执行Flux工作流生成图片
        self.comfyUI.reset_execution_cache()  # 重置缓存，避免历史数据干扰
        logger.info(f"[Predict] 开始执行Flux图像生成工作流")
        self.comfyUI.run_workflow(current_workflow)

        # 获取生成结果
        output_files = self.comfyUI.get_files([OUTPUT_DIR])
        if not output_files:
            raise RuntimeError("[Predict] 工作流执行失败，未生成任何图片")

        logger.info(f"[Predict] 生成完成，输出文件: {output_files}")
        return [Path(file) for file in output_files]