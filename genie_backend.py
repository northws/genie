"""
Genie Backend - 连接GUI和实际功能
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


class GenieBackend:
    """Genie后端处理类"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = os.path.join(self.base_dir, "weights")
        self.evaluations_dir = os.path.join(self.base_dir, "evaluations")
    
    def run_training(self, config_path, gpus=None, resume_path=None, 
                    output_callback=None, progress_callback=None):
        """运行训练"""
        try:
            cmd = [sys.executable, "-m", "genie.train", "-c", config_path]
            
            if gpus:
                cmd.extend(["-g", gpus])
            
            if resume_path:
                cmd.extend(["-r", resume_path])
            
            if output_callback:
                output_callback(f"执行命令: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if output_callback:
                        output_callback(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                return "训练完成！"
            else:
                return f"训练失败，返回码: {process.returncode}"
                
        except Exception as e:
            return f"训练出错: {str(e)}"
    
    def run_sampling(self, model_type, model_path, min_length, max_length,
                    batch_size, num_batches, noise_scale, gpu, output_dir,
                    save_trajectory=False, output_callback=None, progress_callback=None):
        """运行采样"""
        try:
            # 解析模型路径
            if model_type == "pretrained":
                # 从预训练模型名称解析路径
                model_name, epoch_str = self._parse_pretrained_model(model_path)
                rootdir = os.path.join(self.base_dir, "weights")
                model_version = 0
                epoch = int(epoch_str.replace("epoch=", ""))
            else:
                # 自定义模型
                model_dir = os.path.dirname(model_path)
                model_name = os.path.basename(model_dir)
                rootdir = os.path.dirname(model_dir)
                model_version = 0
                epoch = self._extract_epoch_from_ckpt(model_path)
            
            cmd = [
                sys.executable, "-m", "genie.sample",
                "-r", rootdir,
                "-n", model_name,
                "-v", str(model_version),
                "-e", str(epoch),
                "--batch_size", str(batch_size),
                "--num_batches", str(num_batches),
                "--noise_scale", str(noise_scale),
                "--min_length", str(min_length),
                "--max_length", str(max_length)
            ]
            
            if gpu:
                cmd.extend(["-g", gpu])
            
            if save_trajectory:
                cmd.append("--save_trajectory")
            
            if output_callback:
                output_callback(f"执行命令: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if output_callback:
                        output_callback(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                return "采样完成！"
            else:
                return f"采样失败，返回码: {process.returncode}"
                
        except Exception as e:
            return f"采样出错: {str(e)}"
    
    def run_evaluation(self, input_dir, output_dir, gpus=None,
                      output_callback=None, progress_callback=None):
        """运行评估"""
        try:
            eval_script = os.path.join(self.evaluations_dir, "pipeline", "evaluate.py")
            
            cmd = [
                sys.executable, eval_script,
                "--input_dir", input_dir,
                "--output_dir", output_dir
            ]
            
            if gpus:
                cmd.extend(["-g", gpus])
            
            if output_callback:
                output_callback(f"执行命令: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if output_callback:
                        output_callback(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                return "评估完成！"
            else:
                return f"评估失败，返回码: {process.returncode}"
                
        except Exception as e:
            return f"评估出错: {str(e)}"
    
    def run_plotting(self, plot_type, input_dir, output_dir, input_file=None,
                    output_callback=None, progress_callback=None):
        """运行绘图"""
        try:
            if plot_type in ["单个结构可视化", "轨迹可视化"]:
                # 使用 visualize.py
                vis_script = os.path.join(self.evaluations_dir, "visualize.py")
                cmd = [sys.executable, vis_script, input_file, "-o", output_dir]
            else:
                # 使用 plot.py
                plot_script = os.path.join(self.evaluations_dir, "plot.py")
                cmd = [sys.executable, plot_script, "-i", input_dir, "-o", output_dir]
                
                # 确定绘图类型
                if plot_type == "分析图 (Analysis)":
                    cmd.extend(["-p", "analysis"])
                elif plot_type == "MDS图 (MDS Visualization)":
                    cmd.extend(["-p", "mds"])
                elif plot_type == "结构图 (Structure Examples)":
                    cmd.extend(["-p", "structures"])
                else:  # 全部图表
                    cmd.extend(["-p", "all"])
            
            if output_callback:
                output_callback(f"执行命令: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if output_callback:
                        output_callback(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                return "绘图完成！"
            else:
                return f"绘图失败，返回码: {process.returncode}"
                
        except Exception as e:
            return f"绘图出错: {str(e)}"
    
    def _parse_pretrained_model(self, model_name):
        """解析预训练模型名称"""
        # 例如: "scope_l_128 (epoch=49999)"
        parts = model_name.split(" (")
        name = parts[0]
        epoch = parts[1].rstrip(")")
        return name, epoch
    
    def _extract_epoch_from_ckpt(self, ckpt_path):
        """从检查点文件名提取epoch"""
        filename = os.path.basename(ckpt_path)
        # 例如: epoch=49999.ckpt
        if "epoch=" in filename:
            epoch_str = filename.split("epoch=")[1].split(".")[0]
            return int(epoch_str)
        return 0
    
    def download_dataset(self, dataset_type, output_callback=None):
        """下载数据集"""
        try:
            if dataset_type == "SCOPE":
                script = os.path.join(self.base_dir, "scripts", "install_dataset.sh")
            elif dataset_type == "SwissProt":
                script = os.path.join(self.base_dir, "scripts", "install_dataset.sh")
            else:
                return "未知的数据集类型"
            
            if not os.path.exists(script):
                return f"数据集安装脚本不存在: {script}"
            
            if output_callback:
                output_callback(f"正在下载{dataset_type}数据集...\n")
            
            # 在Windows上可能需要使用bash或git bash
            process = subprocess.Popen(
                ["bash", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if output_callback:
                        output_callback(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                return "数据集下载完成！"
            else:
                return f"数据集下载失败，返回码: {process.returncode}"
                
        except Exception as e:
            return f"数据集下载出错: {str(e)}"
    
    def get_available_models(self):
        """获取可用的预训练模型"""
        models = []
        if os.path.exists(self.weights_dir):
            for model_dir in os.listdir(self.weights_dir):
                model_path = os.path.join(self.weights_dir, model_dir)
                if os.path.isdir(model_path):
                    # 查找检查点文件
                    for f in os.listdir(model_path):
                        if f.endswith('.ckpt'):
                            epoch = f.replace('epoch=', '').replace('.ckpt', '')
                            models.append(f"{model_dir} (epoch={epoch})")
        return models
