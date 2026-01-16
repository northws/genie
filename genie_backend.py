"""Genie Backend - 连接GUI和实际功能
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Windows 标志：阻止创建新控制台窗口
CREATE_NO_WINDOW = 0x08000000


class GenieBackend:
    """Genie后端处理类"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = os.path.join(self.base_dir, "weights")
        self.evaluations_dir = os.path.join(self.base_dir, "evaluations")
    
    def run_training(self, config_path, gpus=None, resume_path=None, 
                    output_callback=None, progress_callback=None, stop_callback=None):
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
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
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
                    save_trajectory=False, output_callback=None, progress_callback=None, stop_callback=None):
        """运行采样"""
        try:
            # 构建采样命令
            cmd = [
                sys.executable, "-m", "genie.sample",
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
            
            # 解析模型路径 - 直接使用ckpt文件路径
            if model_type == "pretrained":
                # 从预训练模型名称解析权重文件夹
                model_name, epoch_str = self._parse_pretrained_model(model_path)
                # 确保包含 .ckpt 后缀
                if not epoch_str.endswith('.ckpt'):
                    epoch_str = f"{epoch_str}.ckpt"
                ckpt_path = os.path.join(self.base_dir, "weights", model_name, epoch_str)
            else:
                # 自定义模型：model_path 应该是 ckpt 文件的完整路径
                ckpt_path = model_path
                model_name = os.path.basename(os.path.dirname(ckpt_path))
            
            # 使用配置文件和检查点
            config_path = os.path.join(self.base_dir, "weights", model_name, "configuration")
            cmd.extend(["--ckpt", ckpt_path, "-c", config_path])
            
            if output_callback:
                output_callback(f"检查点路径: {ckpt_path}\n")
                output_callback(f"配置路径: {config_path}\n")
                output_callback(f"执行命令: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
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
                      output_callback=None, progress_callback=None, stop_callback=None):
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
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
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
                    output_callback=None, progress_callback=None, stop_callback=None):
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
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
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
    
    def download_dataset(self, dataset_type, output_callback=None, progress_callback=None, stop_callback=None):
        """下载数据集"""
        try:
            if output_callback:
                output_callback(f"正在准备下载{dataset_type}数据集...\n")
            
            # 创建data目录
            data_dir = os.path.join(self.base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # 配置SSL证书验证（禁用以解决证书问题）
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            if dataset_type == "SCOPE":
                if output_callback:
                    output_callback("下载SCOPE数据集...\n")
                
                # 下载序列文件
                import urllib.request
                seq_url = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
                seq_file = os.path.join(data_dir, "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
                
                if output_callback:
                    output_callback(f"下载序列文件: {seq_url}\n")
                with urllib.request.urlopen(seq_url, context=ssl_context) as response:
                    with open(seq_file, 'wb') as f:
                        f.write(response.read())
                
                # 下载结构文件
                struct_url = "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz"
                struct_file = os.path.join(data_dir, "pdbstyle-sel-gs-bib-40-2.08.tgz")
                
                if output_callback:
                    output_callback(f"下载结构文件: {struct_url}\n")
                with urllib.request.urlopen(struct_url, context=ssl_context) as response:
                    with open(struct_file, 'wb') as f:
                        f.write(response.read())
                
                # 解压
                if output_callback:
                    output_callback("解压文件...\n")
                import tarfile
                with tarfile.open(struct_file, 'r:gz') as tar:
                    tar.extractall(data_dir)
                
                os.remove(struct_file)
                
                # 预处理
                if output_callback:
                    output_callback("预处理数据集...\n")
                
                script_path = os.path.join(self.base_dir, "scripts", "generate_scope_coords.py")
                if os.path.exists(script_path):
                    result = subprocess.run([sys.executable, script_path], 
                                          capture_output=True, text=True)
                    if output_callback:
                        output_callback(result.stdout)
                
                if output_callback:
                    output_callback("SCOPE数据集下载完成！\n")
                return "SCOPE数据集下载完成！"
            
            else:
                return "暂不支持此数据集类型"
                
        except Exception as e:
            error_msg = f"数据集下载出错: {str(e)}"
            if output_callback:
                output_callback(error_msg)
            return error_msg
    
    def setup_evaluation(self, output_callback=None, progress_callback=None, stop_callback=None):
        """设置评估环境"""
        try:
            if output_callback:
                output_callback("正在设置评估环境...\n")
            
            # 创建packages目录
            packages_dir = os.path.join(self.base_dir, "packages")
            os.makedirs(packages_dir, exist_ok=True)
            
            # 克隆ProteinMPNN
            proteinmpnn_dir = os.path.join(packages_dir, "ProteinMPNN")
            if not os.path.exists(proteinmpnn_dir):
                if output_callback:
                    output_callback("克隆ProteinMPNN仓库...\n")
                subprocess.run(["git", "clone", "https://github.com/dauparas/ProteinMPNN.git", proteinmpnn_dir],
                             check=True)
            
            # 安装ESMFold
            if output_callback:
                output_callback("安装ESMFold...\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "fair-esm[esmfold]"],
                         check=True)
            
            # 安装其他依赖
            if output_callback:
                output_callback("安装评估依赖...\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "modelcif"],
                         check=True)
            
            if output_callback:
                output_callback("评估环境设置完成！\n")
            return "评估环境设置完成！"
            
        except Exception as e:
            error_msg = f"评估环境设置失败: {str(e)}"
            if output_callback:
                output_callback(error_msg)
            return error_msg
    
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
