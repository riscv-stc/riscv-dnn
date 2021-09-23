import numpy as np
import os
import random
import subprocess
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import decode_predictions
# from keras.utils import plot_model
from keras import backend as K

import sys
from PyQt5.QtCore import QObject, QThread, QRect, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPainter

predict_file = open("data/predict.txt", "w")

class InferenceThread(QObject):
    SignalFinished = pyqtSignal()
    SignalResult = pyqtSignal(list)

    def __init__(self, widgetMain, parent=None):
        super(InferenceThread, self).__init__(parent)

    def work(self):
        image_names=sorted(os.listdir('data'))

        keras_pred = []
        r = open('tf_result-fp16.txt', 'r')
        while True:
            line = r.readline()
            if len(line) == 0:
                break
            keras_pred.append(line.split(','))
        r.close()
                
        #print(keras_pred)

        #plt.figure()
        total = 0
        correct = 0
        with subprocess.Popen(["python test_resnet50.py"], shell=True, stdout=subprocess.PIPE) as proc:
            while True:
                line = str(proc.stdout.readline(), encoding="utf-8")

                if len(line) == 0 and proc.poll() is not None:
                    break
        # with subprocess.Popen(["pwd"], shell=True, stdout=subprocess.PIPE) as proc:
        #     while True:
        #         line = "result 0 0,2,0,0,3,21,28,137,0,0,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,28,10,3,4,2,0,0,127,369,3,5,0,37,64,0,89,26,11,831,36,2,9,1042,553,2025,0,3535,4539,7779,51,2629,913,9145,123,5123,4483,9303,4856,99,15268,5215,4377,3871,1,0,1,0,0,0,0,0,0,0,4,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,3,4,0,1,4,0,0,16,0,0,6,33,7,0,6,0,0,0,0,1,3,0,0,0,1,2,0,0,0,0,0,1,1,0,0,0,0,0,0,1,4,1,1,1,0,0,1,3,0,0,33,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,2,0,0,1,4,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,39,1,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,194,0,265,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,3,21,0,0,0,0,3,1,4,2,0,1,1,0,0,4,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,9,3,0,0,6,2746,2,0,0,15,162,0,13,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,5,0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,4,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,3,1,0,0,0,0,0,0,0,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,41,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,6,2,1,332,19,1,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                # line = "result 0 0,2,0,0,3,21,29,140,0,0,18,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,18,30,10,3,4,2,0,0,131,376,3,5,0,39,68,0,93,27,11,870,38,2,10,1109,598,2076,0,3547,4551,7753,52,2567,971,9191,128,5139,4472,9303,4927,102,15268,5189,4347,3899,1,0,1,0,0,0,0,0,0,0,4,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,4,4,0,1,5,0,0,17,0,0,6,35,7,0,6,0,0,0,0,1,3,0,0,0,2,2,0,0,0,0,0,1,1,0,1,0,0,0,0,1,4,1,1,1,0,0,1,3,0,0,35,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,0,0,1,1,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,2,0,0,1,4,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,41,1,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,0,263,0,0,0,0,0,0,0,1,0,0,2,0,0,1,1,3,23,0,0,0,1,4,1,4,2,1,1,1,0,0,4,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,10,3,0,0,7,2759,3,0,0,16,164,0,14,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,5,0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,4,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,3,1,0,0,0,0,0,0,0,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,43,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,6,2,1,352,20,1,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                if "result" not in line:
                    continue
                print(line)

                res, index, data = line.split(' ', 2)
                index = int(index)
                result = np.fromstring(data[:-1], dtype=np.uint16, sep=',')
                result.dtype = np.float16
                result = result.astype("float")
                # print(result)
                result = result[1:1001]
                #result.dtype = "float16"
                result = np.expand_dims(result, 0)
                # result = result.reshape(1, 1000)
                # print("=>", result)
                pred_dec = decode_predictions(result)
                #print(index)
                print(pred_dec[0][0][0] + ',' + pred_dec[0][0][1] + ',' + str(pred_dec[0][0][2]), end=",")
                print(pred_dec[0][1][0] + ',' + pred_dec[0][1][1] + ',' + str(pred_dec[0][1][2]))
           
                if keras_pred[index][1] == pred_dec[0][0][1]:
                    correct += 1
                total += 1
                #print(str(correct) + '/' + str(total) + ": ", str(correct/total*100) + "%" +
                #     ', Keras result: ' + keras_pred[index][1] + ' ' + str(keras_pred[index][2]).strip() +
                #     ', RVM result: ' + pred_dec[0][0][1] + ' ' + str(pred_dec[0][0][2])
                #     )

                if keras_pred[index][1] == pred_dec[0][0][1]: #index+1
                    ok = "O"
                else:
                    ok = "X"
                print(f'%4d/%4d=%3.2f %s, Keras result: %s => %s=%s, RVM result: %s=%s' % (
                    correct, total, correct/total*100, ok,
                    keras_pred[index][0], keras_pred[index][1], str(keras_pred[index][2]).strip(),
                    pred_dec[0][0][1],  str(pred_dec[0][0][2])
                    ), file=predict_file)

                kscore = str(keras_pred[index][2])
                fscore = str(pred_dec[0][0][2])
                predict_result = ['dataset/'+image_names[index], keras_pred[index][1], pred_dec[0][0][1], kscore, fscore]
                
                self.SignalResult.emit(predict_result)
        self.SignalFinished.emit()

class Picture(QLabel):
    def __init__(self, parent=None):
        super(Picture,self).__init__(parent)
        self.setMinimumSize(28, 28)
        self.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(255, 100, 0);')
        self.image = QImage()

    def paintEvent(self, evt):
        painter = QPainter(self)
        painter.drawImage(QRect(0,0,self.width(), self.height()), self.image)

    def setImage(self, img):
        self.image = img
        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.__total = 0
        self.__correct = 0
        self.thread = QThread()
        self.setUi()
        self.thread.start()

    def setUi(self):
        self.predictWorker = InferenceThread(self)

        self.predictWorker.moveToThread(self.thread)
        self.predictWorker.SignalFinished.connect(self.thread.quit)
        self.predictWorker.SignalResult.connect(self.addResult)
        self.thread.started.connect(self.predictWorker.work)

        self.setWindowTitle("RVM Demo")
        self.setGeometry(200, 200, 800, 480)

        self.textLabel = QLabel(self)
        self.textLabel.setText("Loading...")
        self.textLabel.setFixedWidth(200)
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.textLabel.setStyleSheet("font-size: 36px; font-weight: bold; color:#fcfcfc; background-color: #333333")

        self.statLabel = QLabel(self)
        self.statLabel.setText("Loading...")
        self.statLabel.setFixedWidth(200)
        self.statLabel.setAlignment(Qt.AlignCenter)
        self.statLabel.setStyleSheet("font-size: 28px; color:#fcfcfc; background-color: #333333")

        self.currLabel = QLabel(self)
        self.currLabel.setFixedWidth(200)
        self.currLabel.setAlignment(Qt.AlignCenter)
        self.currLabel.setStyleSheet("color:#fcfcfc; background-color: #333333")

        self.picture = Picture(self)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.textLabel)
        rightLayout.addWidget(self.statLabel)
        rightLayout.addWidget(self.currLabel)

        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.picture, 1)
        layout.addLayout(rightLayout)

        self.setLayout(layout)
    
    def addResult(self, list):
        path, kres, fres, kscore, fscore = list
        image = QImage()
        image.load(path)
        self.picture.setImage(image)

        if kres == fres:
            self.__correct += 1
            result = "[SAME]"
            self.currLabel.setStyleSheet("font-size: 20px; color:white; background-color: #4caf50")

        else:
            result = "[DIFF]"
            self.currLabel.setStyleSheet("font-size: 20px; color:white; background-color: #ff5722")

        self.__total += 1

        percent = "%.2f" % (100 * self.__correct / self.__total)

        self.textLabel.setText(f"{percent}%")
        self.statLabel.setText(f"{self.__correct}/{self.__total}")
        #self.currLabel.setText(f"{kres}:{kscore}{fres}:{fscore}\n\n{result}")
        self.currLabel.setText(f"{kres}\n{fres}\n\n{result}")


if __name__ == '__main__':
    #main()

    app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(800, 480)
    window.show()

    sys.exit(app.exec_())

