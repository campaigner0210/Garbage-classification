import sys
import os
import json
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from garbage import Ui_GarbageSortingUI
import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Main(QMainWindow, Ui_GarbageSortingUI):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.ProjectPath = os.getcwd() + '/垃圾图片库'  # 获取当前工程文件位置
        self.imgName = ''
        self.ture_type = ''
        self.pre_type = ''
        self.jpg = ''
        self.labels = self.label()
        self.img = ''
        self.model = self.get_model()
        self.val_tf = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
        ])
        self.Input.clicked.connect(self.openImage)
        self.Output.clicked.connect(self.calssify)

    def label(self):
        with open('dir_label.txt', 'r', encoding='utf-8') as f:
            labels = f.readlines()
            labels = list(map(lambda x: x.strip().split('\t'), labels))
            return labels

    def openImage(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(None, "打开图片", self.ProjectPath, "All Files(*)")
        self.ture_type = self.imgName.split('/')[-2]
        self.jpg = QtGui.QPixmap(self.imgName).scaled(self.TestImage.width(), self.TestImage.height())
        self.img = Image.open(self.imgName)
        self.img = self.img.convert('RGB')  # 去除透明通道
        self.TestImage.setPixmap(self.jpg)

    def padding_black(self, img):

        w, h = img.size

        scale = 160. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]]) # 等比压缩

        size_fg = img_fg.size
        size_bg = 160

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def get_model(self):
        model = models.resnet101(pretrained=False)
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, 214)
        model = model.cuda()
        # 加载训练好的模型
        checkpoint = torch.load('model/model_best/best_checkpoint_resnet101.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def softmax(self,x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x, 0)
        return softmax_x

    def calssify(self):
        self.img = self.padding_black(self.img)
        self.img = self.val_tf(self.img)
        self.img = self.img.view((-1, 3, 160, 160))

        self.img = self.img.cuda()

        pred = self.model(self.img)
        pred = pred.data.cpu().numpy()[0]
        score = self.softmax(pred)
        self.pred_id = np.argmax(score)
        self.pre_type = self.labels[self.pred_id][0]
        # self.TestResult.setText(self.pre_type)
        self.TestResult.insertPlainText('预测类型：' + self.pre_type + '\n')
        self.TestResult.insertPlainText('真实类型：' + self.ture_type + '\n')
        # self.textBrowser.setText(self.ture_type)
        # self.textBrowser_2.setText(self.pre_type == self.ture_type)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Main()

    MainWindow.show()
    sys.exit(app.exec_())
