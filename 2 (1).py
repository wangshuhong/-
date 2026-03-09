import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入 PySide6 核心组件
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from PySide6.QtCore import Qt

# 导入 Matplotlib 的 Qt 后端
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class MplCanvas(FigureCanvas):
    """自定义的 Matplotlib 画布组件"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 创建三个垂直排列的子图
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312, sharex=self.ax1)
        self.ax3 = self.fig.add_subplot(313, sharex=self.ax1)
        self.fig.subplots_adjust(hspace=0.4)
        super().__init__(self.fig)


class SpotWelderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("点焊机动态阻抗分析系统")
        self.resize(1000, 800)

        self.csv_filepath = ""
        self.processed_data = None

        self.init_ui()

    def init_ui(self):
        # 主控部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 1. 顶部操作区 ---
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("导入 CSV 文件")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_csv)

        self.btn_plot = QPushButton("绘图并分析")
        self.btn_plot.setMinimumHeight(40)
        self.btn_plot.setEnabled(False)  # 没加载文件前禁用
        self.btn_plot.clicked.connect(self.process_and_plot)

        self.lbl_filepath = QLabel("尚未加载文件")
        self.lbl_filepath.setStyleSheet("color: gray;")

        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_plot)
        top_layout.addWidget(self.lbl_filepath, stretch=1)

        main_layout.addLayout(top_layout)

        # --- 2. 中间画图区 ---
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        main_layout.addWidget(self.canvas, stretch=3)

        # --- 3. 底部数据表 ---
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["采样索引 (Index)", "原始 CH1 (mV)", "真实电压 U (V)", "折算电流 I (A)", "动态阻抗 R (Ω)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table, stretch=2)

    def load_csv(self):
        """加载文件对话框"""
        filepath, _ = QFileDialog.getOpenFileName(self, "选择点焊机采样数据", "", "CSV Files (*.csv)")
        if filepath:
            self.csv_filepath = filepath
            self.lbl_filepath.setText(f"已加载: {filepath}")
            self.lbl_filepath.setStyleSheet("color: green;")
            self.btn_plot.setEnabled(True)

    def process_and_plot(self):
        """核心处理与绘图逻辑"""
        if not self.csv_filepath:
            return

        try:
            # 自动跳过 OWON 的非数据表头
            skip_rows = 0
            with open(self.csv_filepath, 'r', encoding='gbk', errors='ignore') as f:
                for i, line in enumerate(f):
                    if line.startswith('1,'):
                        skip_rows = i
                        break

            # 读取数据
            df_raw = pd.read_csv(
                self.csv_filepath,
                encoding='gbk',
                skiprows=skip_rows,
                header=None,
                names=['Index', 'CH1_mV', 'CH2_mV'],
            )

            # --- 硬件与物理参数配置 ---
            U_SCALE = 1.0 / 1000.0
            V_DIVIDER_RATIO = 15.1 / 10.0
            SENSITIVITY = 0.25
            AMPS_PER_MT = 1.0  # 暂时保持 1.0

            # --- 提取与计算 ---
            ch2_zero = df_raw['CH2_mV'][:100].mean()

            # 基于前100点噪声自适应阈值（并保留80mV下限，兼容旧逻辑）
            ch2_baseline = df_raw['CH2_mV'][:100]
            ch2_sigma = ch2_baseline.std(ddof=0)
            pulse_threshold = max(80.0, 6.0 * ch2_sigma)

            pulse_mask = np.abs(df_raw['CH2_mV'] - ch2_zero) > pulse_threshold
            active_idx = df_raw.index[pulse_mask]

            if len(active_idx) == 0:
                QMessageBox.warning(self, "警告", "未检测到有效的电流脉冲，请检查数据。")
                return

            pulse_start_idx = int(active_idx[0])
            pulse_end_idx = int(active_idx[-1])

            start = max(0, pulse_start_idx - 100)
            end = min(len(df_raw) - 1, pulse_end_idx + 200)
            data = df_raw.iloc[start:end + 1].copy()

            # CH1 是焊点两端绝对电压，不能减去静态基准
            data['U_V'] = data['CH1_mV'] * U_SCALE
            v_diff = (data['CH2_mV'] - ch2_zero) / 1000.0
            data['I_A'] = (v_diff * V_DIVIDER_RATIO / SENSITIVITY) * AMPS_PER_MT

            # 仅在电流脉冲主体内计算阻抗，避免停焊段小电流把 R=U/I 放大到 100Ω+。
            # 这里使用“固定下限 + 峰值比例下限”的组合门限，兼顾不同量程。
            peak_current = data['I_A'].abs().max()
            current_threshold = max(0.1 * AMPS_PER_MT, 0.1 * peak_current)
            in_pulse_window = (data.index >= pulse_start_idx) & (data.index <= pulse_end_idx)
            valid_mask = in_pulse_window & (data['I_A'].abs() > current_threshold)
            data['R_ohm'] = np.nan
            data.loc[valid_mask, 'R_ohm'] = np.abs(data.loc[valid_mask, 'U_V'] / data.loc[valid_mask, 'I_A'])

            self.processed_data = data

            # --- 更新图表 ---
            self.update_canvas()

            # --- 更新表格 ---
            self.update_table()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理数据时发生错误:\n{str(e)}")

    def update_canvas(self):
        """刷新 Matplotlib 画布"""
        data = self.processed_data

        # 清空旧图
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax3.clear()

        # 1. 绘制电压
        self.canvas.ax1.plot(data['Index'], data['U_V'], 'r-')
        self.canvas.ax1.set_title("点焊机输出电压 (U)")
        self.canvas.ax1.set_ylabel("电压 (V)")
        self.canvas.ax1.grid(True, alpha=0.5)

        # 2. 绘制电流
        self.canvas.ax2.plot(data['Index'], data['I_A'], 'b-')
        self.canvas.ax2.set_title("点焊电流释放脉冲 (I)")
        self.canvas.ax2.set_ylabel("电流 (相对A)")
        self.canvas.ax2.grid(True, alpha=0.5)

        # 3. 绘制阻抗
        self.canvas.ax3.plot(data['Index'], data['R_ohm'], 'g-')
        self.canvas.ax3.set_title("动态接触阻抗 (R=U/I)")
        self.canvas.ax3.set_ylabel("阻抗 (Ω)")
        self.canvas.ax3.set_xlabel("采样序列号 (Index)")
        self.canvas.ax3.grid(True, alpha=0.5)

        # 限幅防止无穷大撑爆图表
        data['R_ohm'] = data['R_ohm'].replace([np.inf, -np.inf], np.nan)
        r_clean = data['R_ohm'].dropna()
        if not r_clean.empty:
            max_y = max(r_clean.median() * 5, 0.001)
            self.canvas.ax3.set_ylim(0, max_y)

        self.canvas.draw()

    def update_table(self):
        """将数据填入底部表格"""
        data = self.processed_data
        self.table.setRowCount(len(data))

        for row, (_, row_data) in enumerate(data.iterrows()):
            # 填充每一列数据，格式化小数位数
            idx_item = QTableWidgetItem(str(int(row_data['Index'])))
            ch1_item = QTableWidgetItem(f"{row_data['CH1_mV']:.2f}")
            u_item = QTableWidgetItem(f"{row_data['U_V']:.4f}")
            i_item = QTableWidgetItem(f"{row_data['I_A']:.4f}")

            # 处理 NaN 的显示
            r_val = row_data['R_ohm']
            r_str = f"{r_val:.4f}" if not pd.isna(r_val) else "无效/无穷大"
            r_item = QTableWidgetItem(r_str)

            # 居中对齐
            for item in [idx_item, ch1_item, u_item, i_item, r_item]:
                item.setTextAlignment(Qt.AlignCenter)

            self.table.setItem(row, 0, idx_item)
            self.table.setItem(row, 1, ch1_item)
            self.table.setItem(row, 2, u_item)
            self.table.setItem(row, 3, i_item)
            self.table.setItem(row, 4, r_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpotWelderApp()
    window.show()
    sys.exit(app.exec())
