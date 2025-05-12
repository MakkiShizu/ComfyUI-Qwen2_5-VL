import os
import torch
import folder_paths
import numpy as np
import shutil
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
)
from pathlib import Path
from comfy_api.input import VideoInput

model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)


class DownloadAndLoadQwen2_5_VLModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-32B-Instruct",
                        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-72B-Instruct",
                        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                    ],
                    {"default": "Qwen/Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "8bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
            },
        }

    RETURN_TYPES = ("QWEN2_5_VL_MODEL",)
    RETURN_NAMES = ("Qwen2_5_VL_model",)
    FUNCTION = "DownloadAndLoadQwen2_5_VLModel"
    CATEGORY = "Qwen2_5-VL"

    def DownloadAndLoadQwen2_5_VLModel(self, model, quantization, attention):
        Qwen2_5_VL_model = {"model": "", "model_path": ""}
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading Qwen2.5VL model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model, local_dir=model_path, local_dir_use_symlinks=False
            )
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None
        Qwen2_5_VL_model["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
        )
        Qwen2_5_VL_model["model_path"] = model_path
        return (Qwen2_5_VL_model,)


class Qwen2_5_VL_Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen2_5_VL_model": ("QWEN2_5_VL_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen2_5_VL_Run"
    CATEGORY = "Qwen2_5-VL"

    def Qwen2_5_VL_Run(
        self,
        text,
        Qwen2_5_VL_model,
        video_decode_method,
        max_new_tokens,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
        BatchImage=None,
    ):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28
        processor = AutoProcessor.from_pretrained(Qwen2_5_VL_model["model_path"])
        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    )
        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )
        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
        if text:
            content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen2_5_VL_model["model"].device)
        generated_ids = Qwen2_5_VL_model["model"].generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # delete_temp_image(seed)
        # delete_temp_video(seed)
        return (output_text,)


class BatchImageLoaderToLocalFiles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("BatchImage",)
    RETURN_NAMES = ("BatchImage",)
    FUNCTION = "BatchImageLoaderToLocalFiles"
    CATEGORY = "Qwen2_5-VL"

    def BatchImageLoaderToLocalFiles(self, **kwargs):
        images = list(kwargs.values())
        image_paths = []
        for idx, image in enumerate(images):
            image_path = Path(folder_paths.temp_directory) / f"temp_image_{idx}.png"
            img = Image.fromarray(
                np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            )
            img.save(os.path.join(image_path))
            image_paths.append(f"file://{image_path.resolve().as_posix()}")
        return (image_paths,)


def temp_video(video: VideoInput, seed):
    video_path = Path(folder_paths.temp_directory) / f"temp_video_{seed}.mp4"
    video.save_to(
        os.path.join(video_path),
        format="mp4",
        codec="h264",
    )
    uri = f"{video_path.as_posix()}"
    return uri


# def delete_temp_video(seed):
#     video_path = Path(folder_paths.temp_directory) / f"temp_video_{seed}.mp4"
#     if video_path.exists():
#         video_path.unlink()


def temp_image(image, seed):
    image_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
    img = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    img.save(os.path.join(image_path))
    uri = f"file://{image_path.as_posix()}"

    return uri


def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for Nth_count in range(num_counts):
        img = Image.fromarray(
            np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}.png"
        img.save(os.path.join(image_path))
        image_paths.append(f"file://{image_path.resolve().as_posix()}")
    return image_paths


# def delete_temp_image(seed):
#     image_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
#     multiple_image_path = Path(folder_paths.temp_directory) / "Multiple"
#     if image_path.exists():
#         image_path.unlink()
#     try:
#         shutil.rmtree(multiple_image_path)
#     except FileNotFoundError:
#         pass


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQwen2_5_VLModel": DownloadAndLoadQwen2_5_VLModel,
    "Qwen2_5_VL_Run": Qwen2_5_VL_Run,
    "BatchImageLoaderToLocalFiles": BatchImageLoaderToLocalFiles,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQwen2_5_VLModel": "DownloadAndLoadQwen2_5_VLModel",
    "Qwen2_5_VL_Run": "Qwen2_5_VL_Run",
    "BatchImageLoaderToLocalFiles": "BatchImageLoaderToLocalFiles",
}
