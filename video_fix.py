import cv2
import torch
import time
import numpy as np
from PIL import Image
from src.pipeline_difix import DifixPipeline

def process_video(input_path="input.mp4", output_path="output.mp4", save_debug_frames=True):
    # 读取视频
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    prev_fixed_frame = None
    first_fixed_frame = None
    pipe = None
    pipe_ref = None
    
    print("开始处理视频...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR转RGB，转换为PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        
        if frame_idx == 0:
            # 第一帧：使用无参考模型
            print("加载无参考模型...")
            pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
            pipe.to("cuda")
            
            print(f"处理第{frame_idx + 1}帧（无参考）")
            start_time = time.time()
            fixed_frame = pipe(
                prompt="remove degradation",
                image=pil_frame,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0
            ).images[0]
            process_time = time.time() - start_time
            print(f"第{frame_idx + 1}帧处理完成，耗时：{process_time:.2f}秒")
            
            # 保存第一帧作为质量参考
            first_fixed_frame = fixed_frame.copy()
            
            # 释放无参考模型显存
            del pipe
            torch.cuda.empty_cache()
            
            # 加载参考模型
            print("加载参考模型...")
            pipe_ref = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
            pipe_ref.to("cuda")
            
        else:
            # 后续帧：使用参考模型
            print(f"处理第{frame_idx + 1}帧（参考第{frame_idx}帧）")
            
            # 计算当前帧与参考帧的相似度
            current_array = np.array(pil_frame)
            ref_array = np.array(prev_fixed_frame)
            similarity = np.corrcoef(current_array.flatten(), ref_array.flatten())[0,1]
            print(f"  当前帧与参考帧相似度：{similarity:.4f}")
            
            start_time = time.time()
            fixed_frame = pipe_ref(
                prompt="remove degradation",
                image=pil_frame,
                ref_image=prev_fixed_frame,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0
            ).images[0]
            process_time = time.time() - start_time
            print(f"第{frame_idx + 1}帧处理完成，耗时：{process_time:.2f}秒")
            
            # 计算修复前后的差异
            original_array = np.array(pil_frame)
            fixed_array = np.array(fixed_frame)
            diff_ratio = np.mean(np.abs(fixed_array.astype(float) - original_array.astype(float))) / 255.0
            print(f"  修复前后差异比例：{diff_ratio:.4f}")
            
            # 检测是否偏离第一帧质量
            if frame_idx > 5:  # 从第6帧开始检测
                first_array = np.array(first_fixed_frame)
                quality_drift = np.mean(np.abs(fixed_array.astype(float) - first_array.astype(float))) / 255.0
                print(f"  与第一帧质量偏差：{quality_drift:.4f}")
                if quality_drift > 0.3:  # 阈值可调整
                    print("  ⚠️  警告：质量偏差较大，可能出现错误累积！")
        
        # 转换回OpenCV格式并写入视频
        fixed_array = cv2.cvtColor(np.array(fixed_frame), cv2.COLOR_RGB2BGR)
        out.write(fixed_array)
        
        # 保存调试帧（每10帧或质量异常时）
        if save_debug_frames and (frame_idx % 10 == 0 or frame_idx < 5):
            debug_dir = "debug_frames"
            import os
            os.makedirs(debug_dir, exist_ok=True)
            fixed_frame.save(f"{debug_dir}/frame_{frame_idx:04d}_fixed.png")
            pil_frame.save(f"{debug_dir}/frame_{frame_idx:04d}_original.png")
            print(f"  已保存调试帧：{debug_dir}/frame_{frame_idx:04d}_*.png")
        
        # 保存当前修复帧作为下一帧的参考
        prev_fixed_frame = fixed_frame
        frame_idx += 1
    
    # 释放资源
    cap.release()
    out.release()
    if pipe_ref is not None:
        del pipe_ref
        torch.cuda.empty_cache()
    
    print(f"\n视频处理完成！")
    print(f"总共处理了{frame_idx}帧")
    print(f"输出文件：{output_path}")

if __name__ == "__main__":
    process_video()