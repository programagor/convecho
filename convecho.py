import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QCheckBox
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy.signal import fftconvolve
from mpl_toolkits.mplot3d import Axes3D

class FunctionPlotter(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Convolution and Matched Filter simulation')

        main_layout = QVBoxLayout()

        # Chirp parameters layout
        chirp_layout = QVBoxLayout()
        chirp_layout.addWidget(QLabel('<h3>Chirp Parameters</h3>'))

        self.resolution = 1000

        self.slider_duration = QSlider(Qt.Horizontal)
        self.slider_duration.setRange(10, 100)
        self.slider_duration.setValue(20)
        self.slider_duration.valueChanged.connect(self.update_plots)
        chirp_layout.addWidget(QLabel('Duration'))
        chirp_layout.addWidget(self.slider_duration)

        self.slider_f_start = QSlider(Qt.Horizontal)
        self.slider_f_start.setRange(1, 50)
        self.slider_f_start.setValue(10)
        self.slider_f_start.valueChanged.connect(self.update_plots)
        chirp_layout.addWidget(QLabel('Start Frequency'))
        chirp_layout.addWidget(self.slider_f_start)

        self.slider_f_end = QSlider(Qt.Horizontal)
        self.slider_f_end.setRange(1, 100)
        self.slider_f_end.setValue(30)
        self.slider_f_end.valueChanged.connect(self.update_plots)
        chirp_layout.addWidget(QLabel('End Frequency'))
        chirp_layout.addWidget(self.slider_f_end)

        # Echo parameters layout
        echo_layout = QVBoxLayout()
        echo_layout.addWidget(QLabel('<h3>Echo Parameters</h3>'))

        self.slider_a1 = QSlider(Qt.Horizontal)
        self.slider_a1.setRange(0, 20)
        self.slider_a1.setValue(8)
        self.slider_a1.valueChanged.connect(self.update_plots)
        echo_layout.addWidget(QLabel('Echo 1 Amplitude'))
        echo_layout.addWidget(self.slider_a1)

        self.slider_t1 = QSlider(Qt.Horizontal)
        self.slider_t1.setRange(0, 100)
        self.slider_t1.setValue(20)
        self.slider_t1.valueChanged.connect(self.update_plots)
        echo_layout.addWidget(QLabel('Echo 1 Delay'))
        echo_layout.addWidget(self.slider_t1)

        self.slider_a2 = QSlider(Qt.Horizontal)
        self.slider_a2.setRange(0, 20)
        self.slider_a2.setValue(0)
        self.slider_a2.valueChanged.connect(self.update_plots)
        echo_layout.addWidget(QLabel('Echo 2 Amplitude'))
        echo_layout.addWidget(self.slider_a2)

        self.slider_t2 = QSlider(Qt.Horizontal)
        self.slider_t2.setRange(0, 100)
        self.slider_t2.setValue(40)
        self.slider_t2.valueChanged.connect(self.update_plots)
        echo_layout.addWidget(QLabel('Echo 2 Delay'))
        echo_layout.addWidget(self.slider_t2)

        # Noise parameters layout
        noise_layout = QVBoxLayout()
        noise_layout.addWidget(QLabel('<h3>Noise Parameters</h3>'))

        self.slider_noise = QSlider(Qt.Horizontal)
        self.slider_noise.setRange(0, 100)
        self.slider_noise.setValue(10)
        self.slider_noise.valueChanged.connect(self.update_plots)
        noise_layout.addWidget(QLabel('Noise Amplitude'))
        noise_layout.addWidget(self.slider_noise)

        self.btn_regenerate_noise = QPushButton('Regenerate Noise')
        self.btn_regenerate_noise.clicked.connect(self.regenerate_noise)
        noise_layout.addWidget(self.btn_regenerate_noise)

        # Position and diagonal line
        pos_layout = QVBoxLayout()
        self.slider_pos = QSlider(Qt.Horizontal)
        self.slider_pos.setRange(0, 1000)
        self.slider_pos.setValue(0)
        self.slider_pos.valueChanged.connect(self.update_send_with_position)
        pos_layout.addWidget(QLabel('Position'))
        pos_layout.addWidget(self.slider_pos)

        self.checkbox_diag = QCheckBox('Enable diagonal line')
        self.checkbox_diag.setChecked(True)
        self.checkbox_diag.stateChanged.connect(self.update_plots)
        pos_layout.addWidget(self.checkbox_diag)

        controls_layout = QHBoxLayout()
        controls_layout.addLayout(chirp_layout)
        controls_layout.addLayout(echo_layout)
        controls_layout.addLayout(noise_layout)

        main_layout.addLayout(controls_layout)
        main_layout.addLayout(pos_layout)

        # Plot layout
        plot_layout = QHBoxLayout()

        self.figure, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 8))
        self.figure.subplots_adjust(hspace=0.5)  # Adjust vertical space between plots
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        self.figure3d = plt.figure(figsize=(10, 5))
        self.ax3d = self.figure3d.add_subplot(111, projection='3d')
        self.canvas3d = FigureCanvas(self.figure3d)
        
        plot_layout.addWidget(self.canvas3d)

        main_layout.addLayout(plot_layout)

        self.setLayout(main_layout)

        self.regenerate_noise()
        self.update_plots()

    def send(self, t, f_start, f_end, duration):
        chirp = np.zeros_like(t)
        mask = (t >= 0) & (t <= duration)
        chirp[mask] = np.sin(2 * np.pi * (f_start + (f_end - f_start) * t[mask] / duration) * t[mask])
        return chirp

    def receive(self, t, send_func, a1, t1, a2, t2, noise, a, f_start, f_end, duration):
        return a1 * send_func(t - t1, f_start, f_end, duration) + \
               a2 * send_func(t - t2, f_start, f_end, duration) + \
               a * noise

    def update_plots(self):
        duration = self.slider_duration.value() / 10
        f_start = self.slider_f_start.value() / 10
        f_end = self.slider_f_end.value() / 10
        a1 = self.slider_a1.value() / 10
        t1 = self.slider_t1.value() / 10
        a2 = self.slider_a2.value() / 10
        t2 = self.slider_t2.value() / 10
        noise_amplitude = self.slider_noise.value() / 100
        pos = self.slider_pos.value() / 100

        self.t = np.linspace(-10, 10, self.resolution)
        self.t_display = self.t[(self.t >= 0) & (self.t <= 10)]
        self.t_display_send = self.t[(self.t >= 0) & (self.t <= duration)]
        self.t_display_receive = self.t[(self.t >= 0) & (self.t <= 10)]

        self.send_signal = self.send(self.t, f_start, f_end, duration)
        self.receive_signal = self.receive(self.t, self.send, a1, t1, a2, t2, self.noise, noise_amplitude, f_start, f_end, duration)

        self.conv_signal = fftconvolve(self.receive_signal, self.send_signal[::-1], mode='same')
        self.shifted_send_signal = np.roll(self.send_signal, int(pos / (self.t[1] - self.t[0])))
        self.mult_signal = self.shifted_send_signal[(self.t >= 0) & (self.t <= 10)] * self.receive_signal[(self.t >= 0) & (self.t <= 10)]

        self.positive_mult_signal = np.where(self.mult_signal > 0, self.mult_signal, 0)
        self.negative_mult_signal = np.where(self.mult_signal < 0, self.mult_signal, 0)

        self.ax1.clear()
        self.ax1.plot(self.t_display, self.shifted_send_signal[(self.t >= 0) & (self.t <= 10)])
        self.ax1.set_title('Send Signal (Chirp)')

        self.ax2.clear()
        self.ax2.plot(self.t_display, self.receive_signal[(self.t >= 0) & (self.t <= 10)], color='orange')
        self.ax2.set_title('Receive Signal')

        self.ax3.clear()
        self.ax3.plot(self.t_display, self.positive_mult_signal, label='Positive', color='green')
        self.ax3.plot(self.t_display, self.negative_mult_signal, label='Negative', color='red')
        self.ax3.fill_between(self.t_display, self.positive_mult_signal, color='green', alpha=0.5)
        self.ax3.fill_between(self.t_display, self.negative_mult_signal, color='red', alpha=0.5)
        self.ax3.set_title('Multiplication Signal')

        self.ax4.clear()
        self.ax4.plot(self.t_display, self.conv_signal[(self.t >= 0) & (self.t <= 10)], color='purple')
        self.vertical_line = self.ax4.axvline(x=pos, color='black', linestyle='--')
        self.intersection_dot = self.ax4.scatter([pos], [self.conv_signal[np.argmin(np.abs(self.t - pos))]], color='blue', zorder=5)
        self.ax4.set_title('Convolution Signal')

        self.ax3d.clear()
        self.update_3d_plot(duration, pos)

        self.canvas.draw()
        self.canvas3d.draw()

    def update_3d_plot(self, duration, pos):
        X, Y = np.meshgrid(self.t_display_receive, self.t_display_send)
        send_display = self.send_signal[(self.t >= 0) & (self.t <= duration)]
        receive_display = self.receive_signal[(self.t >= 0) & (self.t <= 10)]
        Z = np.outer(send_display, receive_display)

        if not self.checkbox_diag.isChecked():
            self.ax3d.clear()
            self.ax3d.contour3D(X, Y, Z, 50, cmap='PiYG')
        else:
            self.truncate_3d_plot(Z, pos)

        self.ax3d.set_xlim(0, 10)
        self.ax3d.set_ylim(0, duration)
        self.ax3d.set_zlim(np.min(Z), np.max(Z))
        self.ax3d.set_title('Outer product of Send and Receive Signals')
        self.ax3d.set_xlabel('Receive Signal (x)')
        self.ax3d.set_ylabel('Send Signal (y)')
        self.ax3d.set_zlabel('Z = Receive(x) * Send(y)')
        self.ax3d.set_box_aspect([10/duration, 1, 0.5])


    def truncate_3d_plot(self, Z, pos):
        duration = self.slider_duration.value() / 10
        X, Y = np.meshgrid(self.t_display_receive, self.t_display_send)

        x_index = np.argmin(np.abs(self.t_display_receive - pos))
        Y_trunc = Y.copy()
        for i in range(len(Y)):
            Y_trunc[i-1, (i + x_index):] = np.nan

        self.ax3d.clear()
        # First contour plot with full opacity
        self.ax3d.contour3D(X, Y_trunc, Z, 50, cmap='PiYG', alpha=1)

        Y_trunc_half = Y.copy()
        for i in range(len(Y)):
            Y_trunc_half[i, :(i + x_index)] = np.nan

        # Second contour plot with half opacity
        self.ax3d.contour3D(X, Y_trunc_half, Z, 50, cmap='PiYG', alpha=0.01)

        diag_x = self.t_display[(self.t_display - pos >= 0) & (self.t_display - pos <= duration)]
        diag_y = diag_x - pos
        diag_z = self.mult_signal[(self.t_display - pos >= 0) & (self.t_display - pos <= duration)]
        diag_z_pos = np.where(diag_z >= 0, diag_z, 0)
        diag_z_neg = np.where(diag_z < 0, diag_z, 0)

        self.ax3d.plot(diag_x, diag_y, diag_z_pos, color='green', linewidth=2, zorder=10, label='Positive Diagonal')
        self.ax3d.plot(diag_x, diag_y, diag_z_neg, color='red', linewidth=2, zorder=9, label='Negative Diagonal')

    def update_send_with_position(self):
        pos = self.slider_pos.value() / 100
        self.shifted_send_signal = np.roll(self.send_signal, int(pos / (self.t[1] - self.t[0])))
        self.mult_signal = self.shifted_send_signal[(self.t >= 0) & (self.t <= 10)] * self.receive_signal[(self.t >= 0) & (self.t <= 10)]

        self.positive_mult_signal = np.where(self.mult_signal > 0, self.mult_signal, 0)
        self.negative_mult_signal = np.where(self.mult_signal < 0, self.mult_signal, 0)

        self.ax1.lines[0].set_ydata(self.shifted_send_signal[(self.t >= 0) & (self.t <= 10)])

        self.ax3.clear()
        self.ax3.plot(self.t_display, self.positive_mult_signal, label='Positive', color='green')
        self.ax3.plot(self.t_display, self.negative_mult_signal, label='Negative', color='red')
        self.ax3.fill_between(self.t_display, self.positive_mult_signal, color='green', alpha=0.5)
        self.ax3.fill_between(self.t_display, self.negative_mult_signal, color='red', alpha=0.5)
        self.ax3.set_title('Multiplication Signal')

        self.vertical_line.set_xdata([pos, pos])
        self.intersection_dot.set_offsets(np.c_[pos, self.conv_signal[np.argmin(np.abs(self.t - pos))]])

        if self.checkbox_diag.isChecked():
            self.ax3d.clear()
            self.update_3d_plot(self.slider_duration.value() / 10, pos)

        self.canvas.draw()
        self.canvas3d.draw()

    def regenerate_noise(self):
        self.noise = np.random.normal(size=self.resolution)
        self.update_plots()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FunctionPlotter()
    ex.show()
    sys.exit(app.exec_())
