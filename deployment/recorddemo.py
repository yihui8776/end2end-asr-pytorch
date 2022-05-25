# -*- coding: utf-8 -*-

import sys
import os
import wave
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.Qt import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from os import path, system, popen
from time import strftime, localtime, time
from sys import argv, exit
from pyaudio import PyAudio, paInt16
from threading import Thread
from PyQt5.QtCore import Qt
import base64
import json
import time
import requests
#from utils.ops import read_wav_bytes

def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = sound.numpy().T
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

def get_audio_length(path):
    output = subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def read_wav_data(filename: str) -> tuple:
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    return wave_data, framerate, num_channel, num_sample_width


def read_wav_bytes(filename: str) -> tuple:
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    return str_data, framerate, num_channel, num_sample_width

class Ui_MainWindow(object):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.win = QMainWindow()
        self.setupUi(self.win)
        self.pause_flag = False
        self.ii = 0
        self.timer = QTimer()
        # self.timer.start(1000)




        self.timer.timeout.connect(self.disciver)
        self.pbt_Record.clicked.connect(self.transcription)
        self.pbt_Play.clicked.connect(self.play_wav)
        self.pbt_Open.clicked.connect(self.open_wav)

        #self.back_btn = QPushButton("选择视频文件")
        #self.label = QLabel("点击选择视频文件开始检测")
        #self.main_layout.addWidget(self.back_btn, 2, 5, 1, 1)
        #self.main_layout.addWidget(self.label, 0, 0, 5, 5)
        #self.label.setAlignment(Qt.AlignCenter)
        #self.back_btn.clicked.connect(self.open_mp4)


    def open_wav(self):
        if path.exists('./media'):
            openpath = ('./media')
        else:
            openpath = os.getcwd()


        fileName, fileType = QFileDialog.getOpenFileName(None, "选取文件",  openpath , "wav Files(*.wav);;All Files(*);")

        base_url = URL = 'http://192.168.1.10:20001/speech'

        #filename = {'wav': open(fileName, 'rb')}
        wav_bytes, sample_rate, channels, sample_width = read_wav_bytes(fileName)
        #r = requests.post(base_url, files=filename)
        datas = {
            'channels': channels,  # 声道数
            'sample_rate': sample_rate,  # 采样频率，每秒多少次采样
            'byte_width': sample_width,  # 采样宽度，量化位数
            'samples': str(base64.urlsafe_b64encode(wav_bytes), encoding='utf-8')  # 数据
        }
        headers = {'Content-Type': 'application/json'}

        t0 = time.time()
        r = requests.post(URL, headers=headers, data=json.dumps(datas))
        t1 = time.time()
        r.encoding = 'utf-8'

        result = json.loads(r.text)
        #self.resText.append(str(result["result"]["sample_rate"]))
        if result["status_code"]== 20000:
            print(result["result"])
            self.text_browser.setText(result["result"])
        else :
            print(result)
            self.text_browser.setText(result)

    def play_wav(self):
        self.pbt_Play.setEnabled(False)
        self.t_play = Thread(target=self.demo)
        self.t_play.start()

    def demo(self):
        if path.exists('./media'):
            self.pbt_Play.setText('播放中..')
            wav_file_path = './media/test.wav'
            wf = wave.open(wav_file_path)
            self.pa = PyAudio()
            stream = self.pa.open(format=self.pa.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                                  rate=wf.getframerate(), output=True)
            chunk = 1024
            print('开始播放')
            data = wf.readframes(chunk)
            while data != b'':
                stream.write(data)
                data = wf.readframes(chunk)
            print('播放完毕')
            # 播放完后将文件关闭
            wf.close()
            # 停止声卡
            stream.stop_stream()
            # 关闭声卡
            stream.close()
            # 终止pyaudio
            self.pa.terminate()
            self.pa = None
            self.pbt_Play.setEnabled(True)
            self.pbt_Play.setText('播放')
        else:
            self.pbt_Play.setEnabled(False)

    def setprogressBar(self, i):
        self.progressBar.setValue(i)

    def disciver(self):
        self.t_discover = Thread(target=self.show_time)
        self.t_discover.start()

    def show_time(self):
        if not path.exists('./media'):
            system('MD media')
        if self.progressBar.value() != self.progressBar.maximum():
            self.ii += 10  #10个线程  ，10秒
            self.progressBar.setValue(self.ii)
        else:
            self.pbt_Play.setEnabled(True)
            self.pause_flag = True

    def transcription(self):
        self.ii = 0
        self.pause_flag = False
        self.progressBar.setValue(0)
        self.pbt_Play.setEnabled(False)
        self.pbt_Record.setEnabled(False)
        self.timer.start(1000)  #1000毫秒 1秒
        self.t_record = Thread(target=self.record)
        self.t_record.start()

    def record(self):
        # os.remove('./media/' + strftime("test", localtime(time())) + '.wav')
        if os.path.exists('./media/test.wav'):
            os.remove('./media/test.wav')
        self.pbt_Record.setText('录制中..')
        # 创建PyAudio对象
        self.pa = PyAudio()
        # 打开声卡，设置 采样深度为16位、声道数为2、采样率为16、模式为输入、采样点缓存数量为2048
        stream = self.pa.open(format=paInt16, channels=2, rate=16000, input=True, frames_per_buffer=2048)
        # 新建一个列表，用来存储采样到的数据
        record_buf = []
        while True:
            if self.pause_flag is True:
                break
            audio_data = stream.read(2048)  # 读出声卡缓冲区的音频数据
            record_buf.append(audio_data)  # 将读出的音频数据追加到record_buf列表
        # my_path = './media/' + strftime("test", localtime(time())) + '.wav'
        my_path = './media/test.wav'
        wf = wave.open(my_path, 'wb')  # 创建一个音频文件
        wf.setnchannels(2)  # 设置声道数为2
        wf.setsampwidth(2)  # 设置采样深度为
        wf.setframerate(16000)  # 设置采样率为16000
        # 将数据写入创建的音频文件
        wf.writeframes("".encode().join(record_buf))
        # 写完后将文件关闭
        wf.close()
        # 停止声卡
        stream.stop_stream()
        # 关闭声卡
        stream.close()
        # 终止pyaudio
        self.pa.terminate()
        self.pa = None
        self.pbt_Record.setText('录制')
        self.pbt_Record.setEnabled(True)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 303)
        MainWindow.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)  # 只显示最小化按钮和关闭按钮
        MainWindow.setFixedSize(self.win.width(), self.win.height())
        #
        # self.main_layout = QGridLayout()
        # self.left_widget = QWidget()
        # self.left_widget.setObjectName('left_widget')
        # self.left_layout = QVBoxLayout()
        # self.left_widget.setLayout(self.left_layout)
        # self.label = QLabel("内容")
        # self.resText = QTextEdit()
        # self.left_layout.addWidget(self.label)
        # self.left_layout.addWidget(self.resText)
        # self.main_layout.addWidget(self.left_widget, 0, 0, 2, 4)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pbt_Record = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_Record.setGeometry(QtCore.QRect(90, 170, 91, 31))
        self.pbt_Record.setObjectName("pbt_Record")

        self.pbt_Play = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_Play.setGeometry(QtCore.QRect(220, 170, 91, 31))
        self.pbt_Play.setObjectName("pbt_Play")
        self.pbt_Play.setEnabled(False)

        self.pbt_Open = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_Open.setGeometry(QtCore.QRect(350, 170, 91, 31))
        self.pbt_Open.setObjectName("pbt_Open")
        self.pbt_Open.setText('选择识别')
        #self.pbt_Open.setEnabled(False)


        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(20, 220, 601, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        #self.browser_label = QLabel('文本输出', self.win)
        self.text_browser = QTextBrowser(self.win)
        self.text_browser.move(40, 40)
        self.text_browser.resize(500, 100)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "recorddemo"))
        self.pbt_Record.setText(_translate("MainWindow", "录制"))
        self.pbt_Play.setText(_translate("MainWindow", "播放"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Ui_MainWindow()
    sys.exit(app.exec_())