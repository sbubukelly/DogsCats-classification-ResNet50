from PyQt5 import QtWidgets, QtGui, QtCore
from torchvision.models.resnet import resnet50

from UI import Ui_MainWindow

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from PIL import Image
import random
from dogcat import Net


# class Net(nn.Module):
#     def __init__(self, model):
#         super(Net, self).__init__()
#         self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
#         self.Linear_layer = nn.Linear(512, 2)

#     def forward(self, x):
#         x = self.resnet_layer(x)
#         x = x.view(x.size(0), -1)
#         x = self.Linear_layer(x)
#         return x


class MainWindow_controller(QtWidgets.QMainWindow):

    resnet = models.resnet50(pretrained=True)
    model = Net(resnet)

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.showModelStructure)
        self.ui.pushButton_3.clicked.connect(self.Result)

    def showModelStructure(self):
        print(self.model)

    def Result(self):
        classes = ('cat', 'dog')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('modelcatdog.pth')
        model = model.to(device)
        model.eval()
        pathDir = os.listdir("./PetImages/Test")
        image = random.sample(pathDir, 1)
        img = cv2.imread("./PetImages/Test/" + image[0])
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        output = model(img)
        prob = F.softmax(output, dim=1)
        # print(prob)
        value, predicted = torch.max(output.data, 1)
        # print(predicted.item())
        # print(value)
        pred_class = classes[predicted.item()]
        print(pred_class)
        img = plt.imread("./PetImages/Test/" + image[0])
        # cv2.imshow(pred_class, img)
        plt.imshow(img)
        plt.title("Class: "+pred_class)
        plt.show()
