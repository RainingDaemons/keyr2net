"""
@Date         : 18-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : load_model.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extra_models.cnn import CNN
from extra_models.lstm import LSTM
from extra_models.cnngru import CNNGRU

from models.keyr2net import KeyR2Net
from models.cnn14 import CNN14
from models.resnet50 import ResNet50
from models.cnneef import CNNEEF
from models.ast import AST
from models.vit import ViT
from models.swin2 import SwinV2

class LoadModel():
    def setup(self, NAME):
        if (NAME == "KEYR2"):
            model = KeyR2Net()
            return model, model.res3
        elif (NAME == "CNN14"):
            model = CNN14()
            return model, model.block6
        elif (NAME == "RESNET50"):
            model = ResNet50()
            return model, model.model.conv1
        elif (NAME == "CNNEEF"):
            model = CNNEEF()
            return model, model.convs
        elif (NAME == "AST"):
            model = AST()
            return model, model.blocks[-1]
        elif (NAME == "VIT"):
            model = ViT()
            return model, model.encoder_layers[-1]
        elif (NAME == "SWIN2"):
            model = SwinV2()
            return model, model.stage3[-1].norm2
        elif (NAME == "CNN"):
            model = CNN()
            return model, model.conv3
        elif (NAME == "LSTM"):
            model = LSTM()
            return model, model.fc
        elif (NAME == "CNNGRU"):
            model = CNNGRU()
            return model, model.conv3
        else:
            return None
