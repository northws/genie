"""Genie Backend - 连接GUI和实际功能
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
    
    def _is_frozen(self):
        """判断是否在打包环境中运行"""
        return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    
    def _get_meipass_path(self, relative_path):
        """获取 PyInstaller 打包后的资源路径"""
        if self._is_frozen():
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(self.base_dir, relative_path)
    
    def _run_with_hidden_window(self, cmd, output_callback=None):
        """运行命令并隐藏窗口"""
        if sys.platform == 'win32':
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                startupinfo=startup_info
            )
        else:
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
            return "任务完成！"
        else:
            return f"任务失败，返回码: {process.returncode}"
    
    def run_training(self, config_path, gpus=None, resume_path=None, 
                    output_callback=None, progress_callback=None, stop_callback=None):
        """运行训练"""
        try:
            if self._is_frozen():
                # 打包环境：直接导入并运行模块
                from genie import train
                
                # 模拟命令行参数
                old_argv = sys.argv
                sys.argv = ['train.py', '-c', config_path]
                if gpus:
                    sys.argv.extend(['-g', gpus])
                if resume_path:
                    sys.argv.extend(['-r', resume_path])
                
                if output_callback:
                    output_callback("开始训练...\n")
                
                # 直接调用训练函数
                train.main()
                
                sys.argv = old_argv
                return "训练完成！"
            else:
                # 开发环境
                train_script = os.path.join(self.base_dir, "genie", "train.py")
                cmd = [sys.executable, train_script, "-c", config_path]
                if gpus:
                    cmd.extend(["-g", gpus])
                if resume_path:
                    cmd.extend(["-r", resume_path])
                
                if output_callback:
                    output_callback(f"执行命令: {' '.join(cmd)}\n")
                
                return self._run_with_hidden_window(cmd, output_callback)
                
        except Exception as e:
            import traceback
            return f"训练出错: {str(e)}\n{traceback.format_exc()}"
    
    def run_sampling(self, model_type, model_path, min_length, max_length,
                    batch_size, num_batches, noise_scale, gpu, output_dir,
                    save_trajectory=False, output_callback=None, progress_callback=None, stop_callback=None):
        """运行采样"""
        try:
            # 解析模型路径
            if model_type == "pretrained":
                model_name, epoch_str = self._parse_pretrained_model(model_path)
                if not epoch_str.endswith('.ckpt'):
                    epoch_str = f"{epoch_str}.ckpt"
                ckpt_path = os.path.join(self.base_dir, "weights", model_name, epoch_str)
            else:
                ckpt_path = model_path
                model_name = os.path.basename(os.path.dirname(ckpt_path))
            
            config_path = os.path.join(self.base_dir, "weights", model_name, "configuration")
            
            if self._is_frozen():
                # 打包环境：直接导入并运行模块
                from genie import sample
                
                old_argv = sys.argv
                sys.argv = ['sample.py', '-c', config_path, '--ckpt', ckpt_path,
                           '--batch_size', str(batch_size), '--num_batches', str(num_batches),
                           '--noise_scale', str(noise_scale), '--min_length', str(min_length),
                           '--max_length', str(max_length)]
                if gpu:
                    sys.argv.extend(['-g', gpu])
                if save_trajectory:
                    sys.argv.append('--save_trajectory')
                
                if output_callback:
                    output_callback(f"检查点路径: {ckpt_path}\n")
                    output_callback(f"配置路径: {config_path}\n")
                    output_callback("开始采样...\n")
                
                sample.main()
                
                sys.argv = old_argv
                return "采样完成！"
            else:
                # 开发环境
                sample_script = os.path.join(self.base_dir, "genie", "sample.py")
                cmd = [sys.executable, sample_script, "-c", config_path, "--ckpt", ckpt_path,
                       "--batch_size", str(batch_size), "--num_batches", str(num_batches),
                       "--noise_scale", str(noise_scale), "--min_length", str(min_length),
                       "--max_length", str(max_length)]
                if gpu:
                    cmd.extend(["-g", gpu])
                if save_trajectory:
                    cmd.append("--save_trajectory")
                
                if output_callback:
                    output_callback(f"检查点路径: {ckpt_path}\n")
                    output_callback(f"配置路径: {config_path}\n")
                
                return self._run_with_hidden_window(cmd, output_callback)
                
        except Exception as e:
            import traceback
            return f"采样出错: {str(e)}\n{traceback.format_exc()}"
    
    def run_evaluation(self, input_dir, output_dir, gpus=None,
                      output_callback=None, progress_callback=None, stop_callback=None):
        """运行评估"""
        try:
            eval_script = os.path.join(self.evaluations_dir, "pipeline", "evaluate.py")
            
            if self._is_frozen():
                # 打包环境：直接运行脚本
                import importlib.util
                spec = importlib.util.spec_from_file_location("evaluate", eval_script)
                evaluate_module = importlib.util.module_from_spec(spec)
                
                old_argv = sys.argv
                sys.argv = ['evaluate.py', '--input_dir', input_dir, '--output_dir', output_dir]
                if gpus:
                    sys.argv.extend(['-g', gpus])
                
                if output_callback:
                    output_callback("开始评估...\n")
                
                spec.loader.exec_module(evaluate_module)
                
                sys.argv = old_argv
                return "评估完成！"
            else:
                cmd = [sys.executable, eval_script, "--input_dir", input_dir, "--output_dir", output_dir]
                if gpus:
                    cmd.extend(["-g", gpus])
                
                if output_callback:
                    output_callback(f"执行命令: {' '.join(cmd)}\n")
                
                return self._run_with_hidden_window(cmd, output_callback)
                
        except Exception as e:
            import traceback
            return f"评估出错: {str(e)}\n{traceback.format_exc()}"
    
    def run_plotting(self, plot_type, input_dir, output_dir, input_file=None,
                    output_callback=None, progress_callback=None, stop_callback=None):
        """运行绘图"""
        try:
            if self._is_frozen():
                # 打包环境
                if plot_type in ["单个结构可视化", "轨迹可视化"]:
                    vis_script = os.path.join(self.evaluations_dir, "visualize.py")
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("visualize", vis_script)
                    vis_module = importlib.util.module_from_spec(spec)
                    
                    old_argv = sys.argv
                    sys.argv = ['visualize.py', input_file or '', '-o', output_dir]
                    
                    if output_callback:
                        output_callback("开始可视化...\n")
                    
                    spec.loader.exec_module(vis_module)
                    sys.argv = old_argv
                    return "可视化完成！"
                else:
                    plot_script = os.path.join(self.evaluations_dir, "plot.py")
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("plot", plot_script)
                    plot_module = importlib.util.module_from_spec(spec)
                    
                    old_argv = sys.argv
                    sys.argv = ['plot.py', '-i', input_dir, '-o', output_dir]
                    
                    if plot_type == "分析图 (Analysis)":
                        sys.argv.extend(["-p", "analysis"])
                    elif plot_type == "MDS图 (MDS Visualization)":
                        sys.argv.extend(["-p", "mds"])
                    elif plot_type == "结构图 (Structure Examples)":
                        sys.argv.extend(["-p", "structures"])
                    else:
                        sys.argv.extend(["-p", "all"])
                    
                    if output_callback:
                        output_callback("开始绘图...\n")
                    
                    spec.loader.exec_module(plot_module)
                    sys.argv = old_argv
                    return "绘图完成！"
            else:
                # 开发环境
                if plot_type in ["单个结构可视化", "轨迹可视化"]:
                    vis_script = os.path.join(self.evaluations_dir, "visualize.py")
                    cmd = [sys.executable, vis_script, input_file or "", "-o", output_dir]
                else:
                    plot_script = os.path.join(self.evaluations_dir, "plot.py")
                    cmd = [sys.executable, plot_script, "-i", input_dir, "-o", output_dir]
                    
                    if plot_type == "分析图 (Analysis)":
                        cmd.extend(["-p", "analysis"])
                    elif plot_type == "MDS图 (MDS Visualization)":
                        cmd.extend(["-p", "mds"])
                    elif plot_type == "结构图 (Structure Examples)":
                        cmd.extend(["-p", "structures"])
                    else:
                        cmd.extend(["-p", "all"])
                
                if output_callback:
                    output_callback(f"执行命令: {' '.join(cmd)}\n")
                
                return self._run_with_hidden_window(cmd, output_callback)
                
        except Exception as e:
            import traceback
            return f"绘图出错: {str(e)}\n{traceback.format_exc()}"
    
    def _parse_pretrained_model(self, model_name):
        """解析预训练模型名称"""
        parts = model_name.split(" (")
        name = parts[0]
        epoch = parts[1].rstrip(")")
        return name, epoch
    
    def download_dataset(self, dataset_type, output_callback=None, progress_callback=None, stop_callback=None):
        """下载数据集"""
        try:
            if output_callback:
                output_callback(f"正在准备下载{dataset_type}数据集...\n")
            
            data_dir = os.path.join(self.base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            if dataset_type == "SCOPE":
                if output_callback:
                    output_callback("下载SCOPE数据集...\n")
                
                import urllib.request
                seq_url = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
                seq_file = os.path.join(data_dir, "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
                
                if output_callback:
                    output_callback(f"下载序列文件...\n")
                with urllib.request.urlopen(seq_url, context=ssl_context) as response:
                    with open(seq_file, 'wb') as f:
                        f.write(response.read())
                
                struct_url = "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz"
                struct_file = os.path.join(data_dir, "pdbstyle-sel-gs-bib-40-2.08.tgz")
                
                if output_callback:
                    output_callback(f"下载结构文件...\n")
                with urllib.request.urlopen(struct_url, context=ssl_context) as response:
                    with open(struct_file, 'wb') as f:
                        f.write(response.read())
                
                if output_callback:
                    output_callback("解压文件...\n")
                import tarfile
                with tarfile.open(struct_file, 'r:gz') as tar:
                    tar.extractall(data_dir)
                
                os.remove(struct_file)
                
                if output_callback:
                    output_callback("SCOPE数据集下载完成！\n")
                return "SCOPE数据集下载完成！"
            
            else:
                return "暂不支持此数据集类型"
                
        except Exception as e:
            import traceback
            error_msg = f"数据集下载出错: {str(e)}\n{traceback.format_exc()}"
            if output_callback:
                output_callback(error_msg)
            return error_msg
    
    def setup_evaluation(self, output_callback=None, progress_callback=None, stop_callback=None):
        """设置评估环境"""
        try:
            if output_callback:
                output_callback("正在设置评估环境...\n")
            
            packages_dir = os.path.join(self.base_dir, "packages")
            os.makedirs(packages_dir, exist_ok=True)
            
            proteinmpnn_dir = os.path.join(packages_dir, "ProteinMPNN")
            if not os.path.exists(proteinmpnn_dir):
                if output_callback:
                    output_callback("克隆ProteinMPNN仓库...\n")
                subprocess.run(["git", "clone", "https://github.com/dauparas/ProteinMPNN.git", proteinmpnn_dir],
                             check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            
            if output_callback:
                output_callback("安装ESMFold...\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "fair-esm[esmfold]"],
                         check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            
            if output_callback:
                output_callback("安装评估依赖...\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "modelcif"],
                         check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            
            if output_callback:
                output_callback("评估环境设置完成！\n")
            return "评估环境设置完成！"
            
        except Exception as e:
            import traceback
            error_msg = f"评估环境设置失败: {str(e)}\n{traceback.format_exc()}"
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
                    for f in os.listdir(model_path):
                        if f.endswith('.ckpt'):
                            epoch = f.replace('epoch=', '').replace('.ckpt', '')
                            models.append(f"{model_dir} (epoch={epoch})")
        return models
