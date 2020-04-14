import mlconfig
from torch import optim

from .rmsprop import TFRMSprop

mlconfig.register(optim.SGD)
mlconfig.register(optim.Adadelta)
mlconfig.register(optim.Adagrad)
mlconfig.register(optim.Adam)
mlconfig.register(optim.AdamW)
mlconfig.register(optim.ASGD)

mlconfig.register(optim.lr_scheduler.MultiStepLR)
mlconfig.register(optim.lr_scheduler.StepLR)
mlconfig.register(optim.lr_scheduler.ExponentialLR)
