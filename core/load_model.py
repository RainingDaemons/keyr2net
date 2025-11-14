"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : load_model.py
"""

from extra_models.cnn import CNN
from extra_models.lstm import LSTM
from extra_models.cnngru import CNNGRU

from models.keyr2net import KeyR2Net
from models.cnn14 import CNN14
from models.resnet50 import ResNet50
from models.cnneef import CNNEEF

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
