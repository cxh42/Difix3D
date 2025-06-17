import cv2
import numpy as np
import torch
from PIL import Image
from src.pipeline_difix import DifixPipeline
from tqdm import tqdm

def resize_to_multiple_of_8(image):
    """调整图像尺寸为8的倍数，避免VAE处理错误"""
    width, height = image.size
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    return image.resize((new_width, new_height), Image.LANCZOS), (width, height)

def process_video(input_path="input.mp4", output_path="output.mp4", use_fp16=True, memory_efficient=True):
    # 初始化模型
    dtype = torch.float16 if use_fp16 else torch.float32
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True, torch_dtype=dtype)
    pipe.to("cuda")
    
    # 内存优化选项（谨慎启用）
    if memory_efficient:
        pipe.enable_vae_slicing()
        # 注意：VAE tiling可能与自定义skip connection冲突，暂时禁用
        try:
            pipe.disable_vae_tiling()  # 确保禁用tiling
        except:
            pass
    
    # 读取视频
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"处理视频: {total_frames} 帧, {fps} FPS, {width}x{height}")
    print(f"精度模式: {'FP16' if use_fp16 else 'FP32'}, 内存优化: {'开启' if memory_efficient else '关闭'}")
    
    # 逐帧处理
    for _ in tqdm(range(total_frames), desc="处理帧"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # BGR转RGB并转换为PIL图像
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 调整尺寸为8的倍数，记录原始尺寸
        resized_image, original_size = resize_to_multiple_of_8(pil_image)
        
        # 伪影修复
        try:
            fixed_image = pipe(
                prompt="remove degradation", 
                image=resized_image, 
                num_inference_steps=1, 
                timesteps=[199], 
                guidance_scale=0.0
            ).images[0]
            
            # 恢复原始尺寸
            fixed_image = fixed_image.resize(original_size, Image.LANCZOS)
            
        except Exception as e:
            print(f"处理帧时出错，使用原始帧: {e}")
            fixed_image = pil_image
        
        # 转换回BGR并写入视频
        fixed_array = np.array(fixed_image)
        bgr_frame = cv2.cvtColor(fixed_array, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
        
        # 清理GPU缓存（内存优化模式）
        if memory_efficient:
            torch.cuda.empty_cache()
    
    # 释放资源
    cap.release()
    out.release()
    print(f"完成! 输出视频: {output_path}")

if __name__ == "__main__":
    # 默认使用半精度和内存优化
    try:
        process_video()
    except Exception as e:
        print(f"内存优化模式失败: {e}")
        print("尝试使用保守模式...")
        process_video(use_fp16=True, memory_efficient=False)
    
    # 如果显存足够，可以使用全精度：
    # process_video(use_fp16=False, memory_efficient=False)