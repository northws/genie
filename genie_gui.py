"""
Genie GUI Application - Fixed Version
修复所有已知问题的版本
"""
import sys
import os
import traceback
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
    QGroupBox, QFormLayout, QMessageBox, QListWidget, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QPixmap

from genie_backend import GenieBackend


class WorkerThread(QThread):
    """后台工作线程"""
    output_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_stopped = False
    
    def stop(self):
        """停止线程"""
        self._is_stopped = True
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs, 
                             output_callback=self.output_signal.emit,
                             progress_callback=self.progress_signal.emit,
                             stop_callback=lambda: self._is_stopped)
            if self._is_stopped:
                self.finished_signal.emit(False, "任务已被用户停止")
            else:
                self.finished_signal.emit(True, result if isinstance(result, str) else "任务完成！")
        except Exception as e:
            if self._is_stopped:
                self.finished_signal.emit(False, "任务已被用户停止")
            else:
                error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
                self.finished_signal.emit(False, error_msg)


class DragDropLineEdit(QLineEdit):
    """支持拖放的输入框"""
    def __init__(self, parent=None, accept_dirs=True, accept_files=True):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.accept_dirs = accept_dirs
        self.accept_files = accept_files
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path) and self.accept_dirs:
                self.setText(path)
            elif os.path.isfile(path) and self.accept_files:
                self.setText(path)


class ConfigEditorDialog(QDialog):
    """配置文件编辑器对话框"""
    def __init__(self, config_path=None, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.init_ui()
        if config_path and os.path.exists(config_path):
            self.load_config()
        else:
            # 提供默认模板
            self.editor.setPlainText(self.get_default_config())
    
    def get_default_config(self):
        """获取默认配置模板"""
        return """# Genie 训练配置文件
io:
  name: my_training
  log_dir: runs
  data_dir: data
  max_n_res: 128
  
training:
  seed: 42
  n_epoch: 50000
  batch_size: 8
  num_workers: 4
  checkpoint_every_n_epoch: 100
  log_every_n_step: 10

model:
  # 模型参数配置
  pass
"""
    
    def init_ui(self):
        self.setWindowTitle("配置文件编辑器")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        # 文件路径
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("文件:"))
        self.path_label = QLabel(self.config_path or "新建配置")
        path_layout.addWidget(self.path_label)
        path_layout.addStretch()
        layout.addLayout(path_layout)
        
        # 编辑器
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Consolas", 10))
        layout.addWidget(self.editor)
        
        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_config)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.editor.setPlainText(content)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置文件失败: {e}")
    
    def save_config(self):
        """保存配置文件"""
        if not self.config_path:
            self.config_path, _ = QFileDialog.getSaveFileName(
                self, "保存配置文件", "", "YAML文件 (*.yml *.yaml)"
            )
        
        if self.config_path:
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    f.write(self.editor.toPlainText())
                QMessageBox.information(self, "成功", "配置文件已保存")
                self.accept()
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存配置文件失败: {e}")


class TrainingTab(QWidget):
    """训练模块标签页"""
    def __init__(self):
        super().__init__()
        self.backend = GenieBackend()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 数据集选择
        dataset_group = QGroupBox("数据集配置")
        dataset_layout = QFormLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["下载SCOPE数据集", "下载SwissProt数据集", "使用自定义数据集"])
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_layout.addRow("数据集:", self.dataset_combo)
        
        # 添加下载按钮
        self.download_btn = QPushButton("下载数据集")
        self.download_btn.clicked.connect(self.download_dataset)
        dataset_layout.addRow("", self.download_btn)
        
        self.custom_dataset_path = DragDropLineEdit(accept_files=False)
        self.custom_dataset_path.setEnabled(False)
        self.custom_dataset_path.setPlaceholderText("拖放目录到此或点击浏览...")
        self.custom_dataset_btn = QPushButton("浏览...")
        self.custom_dataset_btn.setEnabled(False)
        self.custom_dataset_btn.clicked.connect(self.browse_custom_dataset)
        
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(self.custom_dataset_path)
        custom_layout.addWidget(self.custom_dataset_btn)
        dataset_layout.addRow("自定义路径:", custom_layout)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # 训练参数
        params_group = QGroupBox("训练参数")
        params_layout = QFormLayout()
        
        self.config_path = DragDropLineEdit(accept_dirs=False)
        self.config_path.setPlaceholderText("拖放配置文件到此或点击浏览...")
        self.config_btn = QPushButton("浏览...")
        self.config_btn.clicked.connect(self.browse_config)
        self.edit_config_btn = QPushButton("编辑")
        self.edit_config_btn.clicked.connect(self.edit_config)
        self.new_config_btn = QPushButton("新建")
        self.new_config_btn.clicked.connect(self.new_config)
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path)
        config_layout.addWidget(self.config_btn)
        config_layout.addWidget(self.edit_config_btn)
        config_layout.addWidget(self.new_config_btn)
        params_layout.addRow("配置文件:", config_layout)
        
        self.gpus = QLineEdit("0")
        params_layout.addRow("GPU设备:", self.gpus)
        
        self.resume_path = DragDropLineEdit(accept_dirs=False)
        self.resume_path.setPlaceholderText("可选：拖放检查点文件...")
        self.resume_btn = QPushButton("浏览...")
        self.resume_btn.clicked.connect(self.browse_resume)
        resume_layout = QHBoxLayout()
        resume_layout.addWidget(self.resume_path)
        resume_layout.addWidget(self.resume_btn)
        params_layout.addRow("恢复训练:", resume_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 输出日志
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(QLabel("训练日志:"))
        layout.addWidget(self.output)
        
        self.setLayout(layout)
    
    def on_dataset_changed(self, text):
        is_custom = text == "使用自定义数据集"
        self.custom_dataset_path.setEnabled(is_custom)
        self.custom_dataset_btn.setEnabled(is_custom)
        self.download_btn.setEnabled(not is_custom)
    
    def download_dataset(self):
        """下载数据集"""
        dataset_type = self.dataset_combo.currentText()
        if "SCOPE" in dataset_type:
            dataset_name = "SCOPE"
        elif "SwissProt" in dataset_type:
            dataset_name = "SwissProt"
        else:
            return
        
        self.output.append(f"开始下载{dataset_name}数据集...")
        self.download_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        
        # 创建并启动工作线程
        self.worker = WorkerThread(
            self.backend.download_dataset,
            dataset_name
        )
        self.worker.output_signal.connect(self.on_output)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.on_download_finished)
        self.worker.start()
    
    def on_download_finished(self, success, message):
        self.download_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)
    
    def browse_custom_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.custom_dataset_path.setText(path)
    
    def browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML文件 (*.yml *.yaml)")
        if path:
            self.config_path.setText(path)
    
    def edit_config(self):
        """打开配置编辑器"""
        config_path = self.config_path.text()
        if not config_path:
            config_path = None
        
        dialog = ConfigEditorDialog(config_path, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.config_path:
                self.config_path.setText(dialog.config_path)
    
    def new_config(self):
        """创建新配置"""
        dialog = ConfigEditorDialog(None, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.config_path:
                self.config_path.setText(dialog.config_path)
    
    def browse_resume(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择检查点", "", "检查点文件 (*.ckpt)")
        if path:
            self.resume_path.setText(path)
    
    def start_training(self):
        if not self.config_path.text():
            QMessageBox.warning(self, "警告", "请选择配置文件！")
            return
        
        self.output.append("正在启动训练...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        
        # 创建并启动工作线程
        self.worker = WorkerThread(
            self.backend.run_training,
            self.config_path.text(),
            self.gpus.text() if self.gpus.text() else None,
            self.resume_path.text() if self.resume_path.text() else None
        )
        self.worker.output_signal.connect(self.on_output)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
    
    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.output.append("\n正在停止训练...")
    
    def on_output(self, text):
        self.output.append(text)
        self.output.verticalScrollBar().setValue(
            self.output.verticalScrollBar().maximum()
        )
    
    def on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if success else 0)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)


class SamplingTab(QWidget):
    """采样模块标签页"""
    def __init__(self):
        super().__init__()
        self.backend = GenieBackend()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 模型选择
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        self.model_type = QComboBox()
        self.model_type.addItems(["预训练模型", "自定义模型"])
        self.model_type.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addRow("模型类型:", self.model_type)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.backend.get_available_models())
        model_layout.addRow("预训练模型:", self.model_combo)
        
        self.custom_model_path = DragDropLineEdit(accept_dirs=False)
        self.custom_model_path.setEnabled(False)
        self.custom_model_path.setPlaceholderText("拖放检查点文件...")
        self.custom_model_btn = QPushButton("浏览...")
        self.custom_model_btn.setEnabled(False)
        self.custom_model_btn.clicked.connect(self.browse_custom_model)
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(self.custom_model_path)
        custom_layout.addWidget(self.custom_model_btn)
        model_layout.addRow("自定义路径:", custom_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 采样参数
        params_group = QGroupBox("采样参数")
        params_layout = QFormLayout()
        
        self.min_length = QSpinBox()
        self.min_length.setRange(10, 1000)
        self.min_length.setValue(30)
        params_layout.addRow("最小长度:", self.min_length)
        
        self.max_length = QSpinBox()
        self.max_length.setRange(10, 1000)
        self.max_length.setValue(128)
        params_layout.addRow("最大长度:", self.max_length)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100)
        self.batch_size.setValue(8)
        params_layout.addRow("批次大小:", self.batch_size)
        
        self.num_batches = QSpinBox()
        self.num_batches.setRange(1, 1000)
        self.num_batches.setValue(10)
        params_layout.addRow("批次数:", self.num_batches)
        
        self.noise_scale = QDoubleSpinBox()
        self.noise_scale.setRange(0, 10)
        self.noise_scale.setValue(0.5)
        self.noise_scale.setSingleStep(0.1)
        params_layout.addRow("噪声尺度:", self.noise_scale)
        
        self.gpus = QLineEdit("0")
        params_layout.addRow("GPU设备:", self.gpus)
        
        self.save_trajectory = QCheckBox("保存轨迹")
        params_layout.addRow("", self.save_trajectory)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 输出配置
        output_group = QGroupBox("输出配置")
        output_layout = QFormLayout()
        
        self.output_dir = DragDropLineEdit(accept_files=False)
        self.output_dir.setPlaceholderText("拖放输出目录...")
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.browse_output)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(self.output_btn)
        output_layout.addRow("输出目录:", output_dir_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始采样")
        self.start_btn.clicked.connect(self.start_sampling)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_sampling)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 输出日志
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(QLabel("采样日志:"))
        layout.addWidget(self.output)
        
        self.setLayout(layout)
    
    def on_model_type_changed(self, text):
        is_custom = text == "自定义模型"
        self.model_combo.setEnabled(not is_custom)
        self.custom_model_path.setEnabled(is_custom)
        self.custom_model_btn.setEnabled(is_custom)
    
    def browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择检查点文件", "", "检查点文件 (*.ckpt)")
        if path:
            self.custom_model_path.setText(path)
    
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
    
    def start_sampling(self):
        if not self.output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return
        
        self.output.append("正在启动采样...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        
        model_type = "pretrained" if self.model_type.currentText() == "预训练模型" else "custom"
        model_path = self.model_combo.currentText() if model_type == "pretrained" else self.custom_model_path.text()
        
        self.worker = WorkerThread(
            self.backend.run_sampling,
            model_type,
            model_path,
            self.min_length.value(),
            self.max_length.value(),
            self.batch_size.value(),
            self.num_batches.value(),
            self.noise_scale.value(),
            self.gpus.text() if self.gpus.text() else None,
            self.output_dir.text(),
            self.save_trajectory.isChecked()
        )
        self.worker.output_signal.connect(self.on_output)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
    
    def stop_sampling(self):
        if self.worker:
            self.worker.stop()
            self.output.append("\n正在停止采样...")
    
    def on_output(self, text):
        self.output.append(text)
        self.output.verticalScrollBar().setValue(
            self.output.verticalScrollBar().maximum()
        )
    
    def on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if success else 0)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)


class PlottingTab(QWidget):
    """绘图模块标签页"""
    def __init__(self):
        super().__init__()
        self.backend = GenieBackend()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 绘图类型
        type_group = QGroupBox("绘图类型")
        type_layout = QVBoxLayout()
        
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "分析图 (Analysis)",
            "MDS图 (MDS Visualization)",
            "结构图 (Structure Examples)",
            "全部图表"
        ])
        type_layout.addWidget(self.plot_type)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # 输入输出配置
        io_group = QGroupBox("输入输出配置")
        io_layout = QFormLayout()
        
        self.input_dir = DragDropLineEdit(accept_files=False)
        self.input_dir.setPlaceholderText("拖放输入目录...")
        self.input_btn = QPushButton("浏览...")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir)
        input_layout.addWidget(self.input_btn)
        io_layout.addRow("输入目录:", input_layout)
        
        self.input_file = DragDropLineEdit(accept_dirs=False)
        self.input_file.setPlaceholderText("可选：拖放单个结构文件(pdb/cif)...")
        self.input_file_btn = QPushButton("浏览...")
        self.input_file_btn.clicked.connect(self.browse_input_file)
        input_file_layout = QHBoxLayout()
        input_file_layout.addWidget(self.input_file)
        input_file_layout.addWidget(self.input_file_btn)
        io_layout.addRow("结构文件:", input_file_layout)
        
        self.output_dir = DragDropLineEdit(accept_files=False)
        self.output_dir.setPlaceholderText("拖放输出目录...")
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.output_btn)
        io_layout.addRow("输出目录:", output_layout)
        
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始绘图")
        self.start_btn.clicked.connect(self.start_plotting)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_plotting)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 输出日志
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(QLabel("绘图日志:"))
        layout.addWidget(self.output)
        
        self.setLayout(layout)
    
    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if path:
            self.input_dir.setText(path)
    
    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择结构文件", "", "结构文件 (*.pdb *.cif)")
        if path:
            self.input_file.setText(path)
    
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
    
    def start_plotting(self):
        if not self.input_dir.text() and not self.input_file.text():
            QMessageBox.warning(self, "警告", "请选择输入目录或结构文件！")
            return
        
        if not self.output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return
        
        self.output.append("正在启动绘图...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        
        plot_type = self.plot_type.currentText()
        input_dir = self.input_dir.text() if self.input_dir.text() else None
        input_file = self.input_file.text() if self.input_file.text() else None
        
        self.worker = WorkerThread(
            self.backend.run_plotting,
            plot_type,
            input_dir or "",
            self.output_dir.text(),
            input_file
        )
        self.worker.output_signal.connect(self.on_output)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
    
    def stop_plotting(self):
        if self.worker:
            self.worker.stop()
            self.output.append("\n正在停止绘图...")
    
    def on_output(self, text):
        self.output.append(text)
        self.output.verticalScrollBar().setValue(
            self.output.verticalScrollBar().maximum()
        )
    
    def on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if success else 0)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)


class EvaluationTab(QWidget):
    """评估模块标签页"""
    def __init__(self):
        super().__init__()
        self.backend = GenieBackend()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 评估环境设置
        setup_group = QGroupBox("评估环境")
        setup_layout = QVBoxLayout()
        setup_label = QLabel("首次使用前需要初始化评估环境")
        setup_layout.addWidget(setup_label)
        self.setup_btn = QPushButton("初始化评估环境")
        self.setup_btn.clicked.connect(self.setup_evaluation)
        setup_layout.addWidget(self.setup_btn)
        setup_group.setLayout(setup_layout)
        layout.addWidget(setup_group)
        
        # 输入输出配置
        io_group = QGroupBox("输入输出配置")
        io_layout = QFormLayout()
        
        self.input_dir = DragDropLineEdit(accept_files=False)
        self.input_dir.setPlaceholderText("拖放输入目录...")
        self.input_btn = QPushButton("浏览...")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir)
        input_layout.addWidget(self.input_btn)
        io_layout.addRow("输入目录:", input_layout)
        
        self.output_dir = DragDropLineEdit(accept_files=False)
        self.output_dir.setPlaceholderText("拖放输出目录...")
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.output_btn)
        io_layout.addRow("输出目录:", output_layout)
        
        # 新颖性评估参照目录
        self.ref_dir = DragDropLineEdit(accept_files=False)
        self.ref_dir.setPlaceholderText("可选：参照PDB目录...")
        self.ref_btn = QPushButton("浏览...")
        self.ref_btn.clicked.connect(self.browse_ref)
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.ref_dir)
        ref_layout.addWidget(self.ref_btn)
        io_layout.addRow("参照目录:", ref_layout)
        
        self.gpus = QLineEdit("0")
        io_layout.addRow("GPU设备:", self.gpus)
        
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        # 评估选项
        options_group = QGroupBox("评估选项")
        options_layout = QVBoxLayout()
        
        self.run_folding = QCheckBox("运行折叠预测 (ESMFold)")
        self.run_folding.setChecked(True)
        options_layout.addWidget(self.run_folding)
        
        self.run_inverse_folding = QCheckBox("运行逆向折叠 (ProteinMPNN)")
        self.run_inverse_folding.setChecked(True)
        options_layout.addWidget(self.run_inverse_folding)
        
        self.run_novelty = QCheckBox("运行新颖性评估")
        self.run_novelty.setChecked(True)
        options_layout.addWidget(self.run_novelty)
        
        self.run_tm_score = QCheckBox("计算TM分数")
        self.run_tm_score.setChecked(True)
        options_layout.addWidget(self.run_tm_score)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始评估")
        self.start_btn.clicked.connect(self.start_evaluation)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_evaluation)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 输出日志
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("评估日志:"))
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
    
    def setup_evaluation(self):
        """初始化评估环境"""
        self.output_text.append("开始初始化评估环境...")
        self.setup_btn.setEnabled(False)
        
        self.worker = WorkerThread(
            self.backend.setup_evaluation
        )
        self.worker.output_signal.connect(self.on_output)
        self.worker.finished_signal.connect(self.on_setup_finished)
        self.worker.start()
    
    def on_setup_finished(self, success, message):
        self.setup_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)
    
    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if path:
            self.input_dir.setText(path)
    
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
    
    def browse_ref(self):
        path = QFileDialog.getExistingDirectory(self, "选择参照PDB目录")
        if path:
            self.ref_dir.setText(path)
    
    def start_evaluation(self):
        if not self.input_dir.text():
            QMessageBox.warning(self, "警告", "请选择输入目录！")
            return
        
        if not self.output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return
        
        self.output_text.append("正在启动评估...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.worker = WorkerThread(
            self.backend.run_evaluation,
            self.input_dir.text(),
            self.output_dir.text(),
            self.gpus.text() if self.gpus.text() else None
        )
        self.worker.output_signal.connect(self.on_output)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
    
    def stop_evaluation(self):
        if self.worker:
            self.worker.stop()
            self.output_text.append("\n正在停止评估...")
    
    def on_output(self, text):
        self.output_text.append(text)
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
    
    def on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Genie - 蛋白质结构生成工具 (修复版)")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # 标题
        title = QLabel("Genie 蛋白质结构生成工具")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # 提示标签
        tip = QLabel("💡 修复版：支持拖放、下载数据集、生成配置、评估初始化")
        tip.setStyleSheet("color: green; font-style: italic;")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(tip)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(TrainingTab(), "训练")
        self.tabs.addTab(SamplingTab(), "采样")
        self.tabs.addTab(PlottingTab(), "绘图")
        self.tabs.addTab(EvaluationTab(), "评估")
        
        main_layout.addWidget(self.tabs)
        
        # 状态栏
        self.statusBar().showMessage("就绪 | 修复版 v1.2")
        
        central_widget.setLayout(main_layout)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
