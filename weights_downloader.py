import subprocess
import time
import os
from weights_manifest import WeightsManifest


class WeightsDownloader:
    supported_filetypes = [
        ".ckpt",
        ".safetensors",
        ".sft",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".torchscript",
        ".engine",
        ".patch",
    ]

    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map
        self.comfyui_base_path = "/src/ComfyUI"

    def get_canonical_weight_str(self, weight_str):
        return self.weights_manifest.get_canonical_weight_str(weight_str)

    def get_weights_by_type(self, type):
        return self.weights_manifest.get_weights_by_type(type)

    def download_weights(self, weight_str):
        # 首先检查本地是否已经有这个文件
        if self.check_local_file_exists(weight_str):
            print(f"✅ Using local model: {weight_str}")
            return
        
        # 如果本地没有，再检查官方支持列表
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/replicate/cog-comfyui/blob/main/weights_licenses.md"
                )

            if isinstance(self.weights_map[weight_str], list):
                for weight in self.weights_map[weight_str]:
                    self.download_if_not_exists(
                        weight_str, weight["url"], weight["dest"]
                    )
            else:
                self.download_if_not_exists(
                    weight_str,
                    self.weights_map[weight_str]["url"],
                    self.weights_map[weight_str]["dest"],
                )
        else:
            raise ValueError(
                f"{weight_str} unavailable. View the list of available weights: https://github.com/replicate/cog-comfyui/blob/main/supported_weights.md"
            )

    def check_local_file_exists(self, weight_str):
        """检查本地是否已经有这个模型文件"""
        # 检查标准模型目录
        model_dirs = [
            "models/checkpoints",
            "models/loras", 
            "models/vae",
            "models/controlnet",
            "models/clip",
            "models/clip_vision",
            "models/animatediff_models",
            "models/upscale_models",
            "models/embeddings",
            "models/diffusers",
            "models/unet",
            "models/fooocus",
            "models/diffusion_models",
            "models/text_encoders",
            "models/Joy_caption_two",
            "models/LLM",
            "models/upscale_models"
        ]
        
        for dir_name in model_dirs:
            file_path = os.path.join(self.comfyui_base_path, dir_name, weight_str)
            if os.path.exists(file_path):
                print(f"Found local model at: {file_path}")
                return True
        
        # 检查可能的多级目录结构
        possible_paths = [
            os.path.join(self.comfyui_base_path, "models", weight_str),
            os.path.join(self.comfyui_base_path, weight_str),
            os.path.join("/src", weight_str)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found local model at: {path}")
                return True
        
        return False

    def check_if_file_exists(self, weight_str, dest):
        if dest.endswith(weight_str):
            path_string = dest
        else:
            path_string = os.path.join(dest, weight_str)
        return os.path.exists(path_string)

    def download_if_not_exists(self, weight_str, url, dest):
        if self.check_if_file_exists(weight_str, dest):
            print(f"✅ {weight_str} exists in {dest}")
            return
        WeightsDownloader.download(weight_str, url, dest)

    @staticmethod
    def download(weight_str, url, dest):
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weight_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s")

    def delete_weights(self, weight_str):
        if weight_str in self.weights_map:
            weight_path = os.path.join(self.weights_map[weight_str]["dest"], weight_str)
            if os.path.exists(weight_path):
                os.remove(weight_path)
                print(f"Deleted {weight_path}")