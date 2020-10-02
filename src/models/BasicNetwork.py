# Basic network template, creates self.log and passes config
import logging
from torch import nn

class BasicNetwork(nn.Module):
    def __init__(self, config):
        super(BasicNetwork, self).__init__()
        self.log = logging.getLogger(__name__)
        self.config = config

    def forward(self, x):
        if hasattr(self,"model"):
            return self.model(x)
