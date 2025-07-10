import sys
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QTextEdit, QFrame, QFileDialog, QSizePolicy)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer, Signal
import os
import importlib

# 导入检测模块
import dig_detect
# import pointer_detect_debug
# import pointer_exact

# 基本页面类，所有页面的基类
class BasePage(QWidget):
    # 定义信号用于页面间通信
    switch_page_signal = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI界面，子类必须实现此方法"""
        raise NotImplementedError("子类必须实现setup_ui方法")
    
    def reset_state(self):
        """重置页面状态，返回初始状态"""
        pass  # 默认实现为空，子类可以根据需要重写

# 主页面类
class HomePage(BasePage):
    def setup_ui(self):
        """创建主页面，包含标题和两个选项按钮"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # 标题
        title_label = QLabel("基于rk3588的仪表读取检测系统")
        title_font = QFont("SimHei", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addSpacing(80)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(40)

        # 指针仪表按钮
        pointer_btn = QPushButton("指针仪表")
        pointer_btn.setFont(QFont("SimHei", 14))
        pointer_btn.setMinimumSize(200, 60)
        pointer_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        pointer_btn.clicked.connect(lambda: self.switch_page_signal.emit("pointer"))
        button_layout.addWidget(pointer_btn)

        # 数字仪表按钮
        digital_btn = QPushButton("数字仪表")
        digital_btn.setFont(QFont("SimHei", 14))
        digital_btn.setMinimumSize(200, 60)
        digital_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        digital_btn.clicked.connect(lambda: self.switch_page_signal.emit("digital"))
        button_layout.addWidget(digital_btn)

        layout.addLayout(button_layout)

# 指针仪表页面类
class PointerPage(BasePage):
    def setup_ui(self):
        """创建指针仪表页面"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # 标题
        title_label = QLabel("指针仪表检测")
        title_label.setFont(QFont("SimHei", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addSpacing(80)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(40)
        
        # 粗略测量按钮
        rough_btn = QPushButton("粗略测量")
        rough_btn.setFont(QFont("SimHei", 18))
        rough_btn.setMinimumSize(200, 80)
        rough_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        rough_btn.clicked.connect(self.rough_measurement)
        button_layout.addWidget(rough_btn)
        
        # 精密测量按钮
        precise_btn = QPushButton("精密测量")
        precise_btn.setFont(QFont("SimHei", 18))
        precise_btn.setMinimumSize(200, 80)
        precise_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        precise_btn.clicked.connect(self.precise_measurement)
        button_layout.addWidget(precise_btn)
        
        layout.addLayout(button_layout)
        layout.addSpacing(80)

        # 返回按钮
        back_btn = QPushButton("返回主页")
        back_btn.setFont(QFont("SimHei", 16))
        back_btn.setMinimumSize(200, 60)
        back_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        back_btn.clicked.connect(lambda: self.switch_page_signal.emit("home"))
        layout.addWidget(back_btn, alignment=Qt.AlignCenter)
    
    def rough_measurement(self):
        """粗略测量功能"""
        print("启动粗略测量功能")
        # 运行pointer_detect_debug.py
        import subprocess
        import sys
        import os
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "pointer_detect.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            print(f"启动粗略测量失败: {e}")
    
    def precise_measurement(self):
        """精密测量功能"""
        print("启动精密测量功能")
        # 运行pointer_exact.py
        import subprocess
        import sys
        import os
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "pointer_exactly.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            print(f"启动精密测量失败: {e}")

# 数字仪表页面类
class DigitalPage(BasePage):
    def __init__(self, parent=None):
        # 摄像头相关初始化
        self.cap = None
        self.camera_timer = None
        self.camera_label = None
        super().__init__(parent)
    
    def setup_ui(self):
        """创建数字仪表页面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 标题
        title_label = QLabel("数字仪表")
        title_label.setFont(QFont("SimHei", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title_label)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        main_layout.addLayout(content_layout, stretch=1)

        # 左侧摄像头区域 - 显著增大尺寸
        camera_frame = QFrame()
        camera_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        camera_frame.setMinimumWidth(550)  # 设置更大的最小宽度
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(10, 10, 10, 10)

        # 创建摄像头显示标签
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #111; color: #aaa; font-size: 14px;")
        camera_layout.addWidget(self.camera_label)
        content_layout.addWidget(camera_frame, stretch=10)  # 大幅增加比例

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)  # 设置为最小宽度策略
        content_layout.addWidget(right_panel, stretch=1)

        # 数据信息显示区域
        data_frame = QFrame()
        data_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        data_layout = QVBoxLayout(data_frame)
        data_layout.setContentsMargins(15, 15, 15, 15)

        data_title = QLabel("数据信息")
        data_title.setFont(QFont("SimHei", 16, QFont.Bold))
        data_title.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        data_layout.addWidget(data_title)

        # 创建数据显示区域的水平布局
        data_displays_layout = QHBoxLayout()
        data_displays_layout.setSpacing(10)

        # 实时检测数据显示
        realtime_title = QLabel("实时检测数据")
        realtime_title.setFont(QFont("SimHei", 12))
        realtime_title.setStyleSheet("color: #00ff00; margin-bottom: 5px;")
        realtime_layout = QVBoxLayout()
        realtime_layout.addWidget(realtime_title)
        self.realtime_data_display = QTextEdit()
        self.realtime_data_display.setReadOnly(True)
        self.realtime_data_display.setFont(QFont("SimHei", 12))
        self.realtime_data_display.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border: none; border-radius: 4px;")
        realtime_layout.addWidget(self.realtime_data_display)
        data_displays_layout.addLayout(realtime_layout)

        # 截取画面数据显示
        captured_title = QLabel("截取画面数据")
        captured_title.setFont(QFont("SimHei", 12))
        captured_title.setStyleSheet("color: #ffff00; margin-bottom: 5px;")
        captured_layout = QVBoxLayout()
        captured_layout.addWidget(captured_title)
        self.captured_data_display = QTextEdit()
        self.captured_data_display.setReadOnly(True)
        self.captured_data_display.setFont(QFont("SimHei", 12))
        self.captured_data_display.setStyleSheet("background-color: #1e1e1e; color: #ffff00; border: none; border-radius: 4px;")
        captured_layout.addWidget(self.captured_data_display)
        data_displays_layout.addLayout(captured_layout)

        data_layout.addLayout(data_displays_layout)
        right_layout.addWidget(data_frame, stretch=1)

        # 按钮区域
        button_frame = QFrame()
        button_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(20, 20, 20, 20)
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignCenter)

        # 截取画面按钮
        self.capture_btn = QPushButton("截取画面")
        self.capture_btn.setFont(QFont("SimHei", 14))
        self.capture_btn.setMinimumSize(150, 50)
        self.capture_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        self.capture_btn.clicked.connect(self.capture_frame)
        button_layout.addWidget(self.capture_btn)

        # 继续检测按钮
        self.resume_btn = QPushButton("继续检测")
        self.resume_btn.setFont(QFont("SimHei", 14))
        self.resume_btn.setMinimumSize(150, 50)
        self.resume_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        self.resume_btn.clicked.connect(self.resume_detection)
        self.resume_btn.hide()  # 默认隐藏
        button_layout.addWidget(self.resume_btn)

        # 返回主页按钮
        back_btn = QPushButton("返回主页")
        back_btn.setFont(QFont("SimHei", 14))
        back_btn.setMinimumSize(150, 50)
        back_btn.setStyleSheet("border: 2px solid #666; border-radius: 5px;")
        back_btn.clicked.connect(self.back_to_home)
        button_layout.addWidget(back_btn)

        right_layout.addWidget(button_frame)

    def show_event(self):
        """页面显示时调用"""
        self.start_camera()
        
    def hide_event(self):
        """页面隐藏时调用"""
        self.stop_camera()
        
    def reset_state(self):
        """重置页面状态"""
        # 确保"截取画面"按钮可见，"继续检测"按钮隐藏
        self.capture_btn.show()
        self.resume_btn.hide()
        
        # 清空数据显示区
        self.realtime_data_display.clear()
        self.captured_data_display.clear()

    def back_to_home(self):
        """从数字仪表页面返回主页前，重置UI状态"""
        self.reset_state()
        # 停止摄像头并重置状态
        self.stop_camera()
        # 切换到主页
        self.switch_page_signal.emit("home")

    def capture_frame(self):
        """截取当前画面进行检测"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                result = dig_detect.capture_and_process_frame(frame)
                self.captured_data_display.setText(f"截取画面结果: {result}")
                self.capture_btn.hide()
                self.resume_btn.show()
                self.camera_timer.stop()

    def resume_detection(self):
        """继续视频流检测"""
        if self.camera_timer and not self.camera_timer.isActive():
            self.camera_timer.start(30)
            self.resume_btn.hide()
            self.capture_btn.show()

    def start_camera(self):
        """启动摄像头并开始捕获视频流"""
        # 尝试使用外接摄像头（索引1）
        self.cap = cv2.VideoCapture(1)
        camera_type = "外接摄像头"
        
        # 如果外接摄像头无法打开，尝试使用内置摄像头（索引0）
        if not self.cap.isOpened():
            print("外接摄像头不可用，尝试使用内置摄像头")
            self.cap = cv2.VideoCapture(0)
            camera_type = "内置摄像头"
            
        # 如果两个摄像头都无法打开
        if not self.cap.isOpened():
            self.camera_label.setText("无法打开任何摄像头")
            return
        
        print(f"成功打开{camera_type}")

        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 重置dig_detect模块中的检测状态
        dig_detect.frame_count = 0
        dig_detect.detection_counter = 0
        dig_detect.detection_history = []
        dig_detect.current_display_result = ""
        dig_detect.valid_result = ""
        dig_detect.last_detection = ""
        dig_detect.valid_length = 0
        dig_detect.initial_detection = True

        # 创建定时器用于更新视频帧
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.camera_timer.start(30)  # 约33fps

    def stop_camera(self):
        """停止摄像头捕获并释放资源"""
        if self.camera_timer:
            self.camera_timer.stop()
            self.camera_timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
        # 清空摄像头显示
        if self.camera_label:
            self.camera_label.clear()
            self.camera_label.setText("摄像头已停止")
            
        # 重置dig_detect模块中的检测状态
        dig_detect.frame_count = 0
        dig_detect.detection_counter = 0
        dig_detect.detection_history = []

    def update_frame(self):
        """从摄像头捕获帧并更新UI显示"""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            # 处理实时检测（注意：dig_detect使用BGR格式）
            processed_frame, result = dig_detect.process_realtime_detection(frame)
            if result:
                self.realtime_data_display.setText(f"实时检测结果: {result}")
            
            # 转换BGR为RGB格式用于Qt显示
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            # 转换为QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 调整图像大小以适应标签
            max_size = self.camera_label.size()
            scaled_image = q_image.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 显示图像
            self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def resizeEvent(self, event):
        """窗口大小改变时调整摄像头画面"""
        if self.camera_label and self.cap and self.cap.isOpened():
            self.update_frame()
        super().resizeEvent(event)


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于rk3588的仪表读取检测系统")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
        
    def setup_ui(self):
        """设置主窗口UI"""
        # 创建堆叠窗口用于页面切换
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 创建各个页面
        self.create_pages()
        
        # 连接页面切换信号
        self.connect_page_signals()
        
    def create_pages(self):
        """创建所有页面实例并添加到堆叠窗口"""
        # 创建并添加主页面
        self.home_page = HomePage()
        self.stacked_widget.addWidget(self.home_page)
        
        # 创建并添加指针仪表页面
        self.pointer_page = PointerPage()
        self.stacked_widget.addWidget(self.pointer_page)
        
        # 创建并添加数字仪表页面
        self.digital_page = DigitalPage()
        self.stacked_widget.addWidget(self.digital_page)
        
        # 页面字典，用于通过名称切换页面
        self.pages = {
            "home": self.home_page,
            "pointer": self.pointer_page,
            "digital": self.digital_page
        }
    
    def connect_page_signals(self):
        """连接所有页面的信号"""
        for page_name, page in self.pages.items():
            page.switch_page_signal.connect(self.switch_page)
        
        # 页面切换信号连接到页面变化处理函数
        self.stacked_widget.currentChanged.connect(self.on_page_changed)
    
    def switch_page(self, page_name):
        """切换到指定名称的页面"""
        if page_name in self.pages:
            self.stacked_widget.setCurrentWidget(self.pages[page_name])
    
    def on_page_changed(self, index):
        """页面切换事件处理"""
        current_widget = self.stacked_widget.widget(index)
        previous_widget = self.stacked_widget.currentWidget()
        
        # 如果有前一个页面，且它是DigitalPage类型，调用其hide_event
        if previous_widget:
            if isinstance(previous_widget, DigitalPage):
                previous_widget.hide_event()
        
        # 如果当前页面是DigitalPage类型，调用其show_event
        if isinstance(current_widget, DigitalPage):
            current_widget.show_event()
    
    def resizeEvent(self, event):
        """窗口大小改变事件处理"""
        super().resizeEvent(event)
        
        # 如果当前页面是DigitalPage，调用其resizeEvent
        current_widget = self.stacked_widget.currentWidget()
        if isinstance(current_widget, DigitalPage):
            current_widget.update_frame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())