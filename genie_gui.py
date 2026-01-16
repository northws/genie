"""
Genie GUI Application
集成训练、采样、评估和绘图功能的图形界面应用
"""
import sys
import os
import traceback
from pathlib import Path
import subprocess
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
    QGroupBox, QFormLayout, QMessageBox, QListWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon


class WorkerThread(QThread):
    """后台工作线程，用于执行长时间运行的任务"""
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


class TrainingTab(QWidget):
    """训练模块标签页"""
    def __init__(self):
        super().__init__()
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
        
        self.custom_dataset_path = QLineEdit()
        self.custom_dataset_path.setEnabled(False)
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
        
        self.config_path = QLineEdit()
        self.config_btn = QPushButton("浏览...")
        self.config_btn.clicked.connect(self.browse_config)
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path)
        config_layout.addWidget(self.config_btn)
        params_layout.addRow("配置文件:", config_layout)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(50000)
        params_layout.addRow("训练轮数:", self.epochs)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(8)
        params_layout.addRow("批次大小:", self.batch_size)
        
        self.gpus = QLineEdit("0")
        params_layout.addRow("GPU设备 (如0,1):", self.gpus)
        
        self.resume_path = QLineEdit()
        self.resume_btn = QPushButton("浏览...")
        self.resume_btn.clicked.connect(self.browse_resume)
        resume_layout = QHBoxLayout()
        resume_layout.addWidget(self.resume_path)
        resume_layout.addWidget(self.resume_btn)
        params_layout.addRow("恢复训练 (可选):", resume_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
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
    
    def browse_custom_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.custom_dataset_path.setText(path)
    
    def browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML文件 (*.yml *.yaml)")
        if path:
            self.config_path.setText(path)
    
    def browse_resume(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择检查点", "", "检查点文件 (*.ckpt)")
        if path:
            self.resume_path.setText(path)
    
    def start_training(self):
        # 验证输入
        if not self.config_path.text():
            QMessageBox.warning(self, "警告", "请选择配置文件！")
            return
        
        if self.dataset_combo.currentText() == "使用自定义数据集" and not self.custom_dataset_path.text():
            QMessageBox.warning(self, "警告", "请选择自定义数据集路径！")
            return
        
        self.output.append("正在启动训练...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 这里调用实际的训练函数
        # 为了演示，暂时只显示消息
        self.output.append(f"配置文件: {self.config_path.text()}")
        self.output.append(f"训练轮数: {self.epochs.value()}")
        self.output.append(f"批次大小: {self.batch_size.value()}")
        self.output.append(f"GPU设备: {self.gpus.text()}")


class SamplingTab(QWidget):
    """采样模块标签页"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 模型选择
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        self.model_source = QComboBox()
        self.model_source.addItems(["使用预训练模型", "加载自定义模型"])
        self.model_source.currentTextChanged.connect(self.on_model_source_changed)
        model_layout.addRow("模型来源:", self.model_source)
        
        self.pretrained_model = QComboBox()
        self.pretrained_model.addItems([
            "scope_l_128 (epoch=49999)",
            "scope_l_256 (epoch=29999)",
            "swissprot_l_256 (epoch=99)"
        ])
        model_layout.addRow("预训练模型:", self.pretrained_model)
        
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setEnabled(False)
        self.custom_model_btn = QPushButton("浏览...")
        self.custom_model_btn.setEnabled(False)
        self.custom_model_btn.clicked.connect(self.browse_custom_model)
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(self.custom_model_path)
        custom_layout.addWidget(self.custom_model_btn)
        model_layout.addRow("自定义模型:", custom_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 采样参数
        params_group = QGroupBox("采样参数")
        params_layout = QFormLayout()
        
        self.min_length = QSpinBox()
        self.min_length.setRange(10, 512)
        self.min_length.setValue(50)
        params_layout.addRow("最小长度:", self.min_length)
        
        self.max_length = QSpinBox()
        self.max_length.setRange(10, 512)
        self.max_length.setValue(128)
        params_layout.addRow("最大长度:", self.max_length)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(5)
        params_layout.addRow("批次大小:", self.batch_size)
        
        self.num_batches = QSpinBox()
        self.num_batches.setRange(1, 1000)
        self.num_batches.setValue(2)
        params_layout.addRow("批次数量:", self.num_batches)
        
        self.noise_scale = QDoubleSpinBox()
        self.noise_scale.setRange(0.0, 2.0)
        self.noise_scale.setSingleStep(0.1)
        self.noise_scale.setValue(0.6)
        params_layout.addRow("噪声比例:", self.noise_scale)
        
        self.save_trajectory = QCheckBox()
        params_layout.addRow("保存轨迹:", self.save_trajectory)
        
        self.gpu = QLineEdit("0")
        params_layout.addRow("GPU设备:", self.gpu)
        
        self.output_dir = QLineEdit()
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.output_btn)
        params_layout.addRow("输出目录:", output_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始采样")
        self.start_btn.clicked.connect(self.start_sampling)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 输出日志
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("采样日志:"))
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
    
    def on_model_source_changed(self, text):
        is_custom = text == "加载自定义模型"
        self.pretrained_model.setEnabled(not is_custom)
        self.custom_model_path.setEnabled(is_custom)
        self.custom_model_btn.setEnabled(is_custom)
    
    def browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "检查点文件 (*.ckpt)")
        if path:
            self.custom_model_path.setText(path)
    
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
    
    def start_sampling(self):
        # 验证输入
        if self.model_source.currentText() == "加载自定义模型" and not self.custom_model_path.text():
            QMessageBox.warning(self, "警告", "请选择自定义模型文件！")
            return
        
        if not self.output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return
        
        self.output_text.append("正在启动采样...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)


class EvaluationTab(QWidget):
    """评估模块标签页"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 输入输出配置
        io_group = QGroupBox("输入输出配置")
        io_layout = QFormLayout()
        
        self.input_dir = QLineEdit()
        self.input_btn = QPushButton("浏览...")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir)
        input_layout.addWidget(self.input_btn)
        io_layout.addRow("输入目录:", input_layout)
        
        self.output_dir = QLineEdit()
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.output_btn)
        io_layout.addRow("输出目录:", output_layout)
        
        self.gpus = QLineEdit("0")
        io_layout.addRow("GPU设备 (如0,1):", self.gpus)
        
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
    
    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if path:
            self.input_dir.setText(path)
    
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
    
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


class PlottingTab(QWidget):
    """绘图模块标签页"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 绘图类型选择
        type_group = QGroupBox("绘图类型")
        type_layout = QVBoxLayout()
        
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "分析图 (Analysis)",
            "MDS图 (MDS Visualization)",
            "结构图 (Structure Examples)",
            "全部图表",
            "单个结构可视化",
            "轨迹可视化"
        ])
        self.plot_type.currentTextChanged.connect(self.on_plot_type_changed)
        type_layout.addWidget(self.plot_type)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # 输入输出配置
        io_group = QGroupBox("输入输出配置")
        io_layout = QFormLayout()
        
        self.input_dir = QLineEdit()
        self.input_btn = QPushButton("浏览...")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir)
        input_layout.addWidget(self.input_btn)
        io_layout.addRow("输入目录:", input_layout)
        
        # 单个文件输入（用于结构可视化）
        self.input_file = QLineEdit()
        self.input_file.setEnabled(False)
        self.input_file_btn = QPushButton("浏览...")
        self.input_file_btn.setEnabled(False)
        self.input_file_btn.clicked.connect(self.browse_input_file)
        input_file_layout = QHBoxLayout()
        input_file_layout.addWidget(self.input_file)
        input_file_layout.addWidget(self.input_file_btn)
        io_layout.addRow("输入文件:", input_file_layout)
        
        self.output_dir = QLineEdit()
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
        self.start_btn = QPushButton("生成图表")
        self.start_btn.clicked.connect(self.start_plotting)
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 输出日志
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("绘图日志:"))
        layout.addWidget(self.output_text)
        
        # 结果预览
        self.result_list = QListWidget()
        layout.addWidget(QLabel("生成的图表:"))
        layout.addWidget(self.result_list)
        self.result_list.itemDoubleClicked.connect(self.open_image)
        
        self.setLayout(layout)
    
    def on_plot_type_changed(self, text):
        is_single_file = text in ["单个结构可视化", "轨迹可视化"]
        self.input_dir.setEnabled(not is_single_file)
        self.input_btn.setEnabled(not is_single_file)
        self.input_file.setEnabled(is_single_file)
        self.input_file_btn.setEnabled(is_single_file)
    
    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if path:
            self.input_dir.setText(path)
    
    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "数据文件 (*.npy *.txt *.pdb)")
        if path:
            self.input_file.setText(path)
    
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)
    
    def start_plotting(self):
        plot_type = self.plot_type.currentText()
        
        if plot_type in ["单个结构可视化", "轨迹可视化"]:
            if not self.input_file.text():
                QMessageBox.warning(self, "警告", "请选择输入文件！")
                return
        else:
            if not self.input_dir.text():
                QMessageBox.warning(self, "警告", "请选择输入目录！")
                return
        
        if not self.output_dir.text():
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return
        
        self.output_text.append(f"正在生成{plot_type}...")
        self.start_btn.setEnabled(False)
    
    def open_image(self, item):
        """双击打开图片"""
        filepath = os.path.join(self.output_dir.text(), item.text())
        if os.path.exists(filepath):
            os.startfile(filepath)


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Genie - 蛋白质结构生成工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 标题
        title = QLabel("Genie 蛋白质结构生成工具")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(TrainingTab(), "训练")
        self.tabs.addTab(SamplingTab(), "采样")
        self.tabs.addTab(EvaluationTab(), "评估")
        self.tabs.addTab(PlottingTab(), "绘图")
        
        main_layout.addWidget(self.tabs)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        central_widget.setLayout(main_layout)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
