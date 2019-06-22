# -*-coding:utf-8 -*-
from .lenet import *
from .alexnet import *
from .vgg import *
from .resnet import *
from .preresnet import *
from .densenet import *
from .resnext import *
from .senet import *
from .mobilenetv1 import *
from .mobilenetv2 import *

def get_model(config):
    return globals()[config.architecture](config.num_classes)
