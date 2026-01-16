"""
Genie GUI Application - Enhanced Version
增强版GUI应用，包含更多功能
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
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
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


class ConfigEditorDialog(QDialog):
    """配置文件编辑器对话框"""
    def __init__(self, config_path=None, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.init_ui()
        if config_path and os.path.exists(config_path):
            self.load_config()
    
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


class EnhancedTrainingTab(QWidget):
    """增强的训练模块"""
    def __init__(self):
        super().__init__()
        self.backend = GenieBackend()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 数据集配置
        dataset_group = QGroupBox("数据集配置")
        dataset_layout = QFormLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["下载SCOPE数据集", "下载SwissProt数据集", "使用自定义数据集"])
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_layout.addRow("数据集:", self.dataset_combo)
        
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
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path)
        config_layout.addWidget(self.config_btn)
        config_layout.addWidget(self.edit_config_btn)
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
            self.gpus.text(),
            self.resume_path.text() or None
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
    
    def on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if success else 0)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "失败", message)


class ImagePreviewDialog(QDialog):
    """图片预览对话框"""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(f"预览: {os.path.basename(self.image_path)}")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        # 图片显示
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            # 缩放到适合窗口
            pixmap = pixmap.scaled(780, 550, Qt.AspectRatioMode.KeepAspectRatio, 
                                  Qt.TransformationMode.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
        else:
            layout.addWidget(QLabel("无法加载图片"))
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Genie - 蛋白质结构生成工具 (增强版)")
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
        tip = QLabel("💡 提示：支持拖放文件到输入框")
        tip.setStyleSheet("color: gray; font-style: italic;")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(tip)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(EnhancedTrainingTab(), "训练")
        # 其他标签页可以类似方式增强
        
        main_layout.addWidget(self.tabs)
        
        # 状态栏
        self.statusBar().showMessage("就绪 | 增强版包含：文件拖放、配置编辑器、停止按钮、图片预览")
        
        central_widget.setLayout(main_layout)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
