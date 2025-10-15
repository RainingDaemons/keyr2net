"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : load_model.py
"""

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
        else:
            return None
