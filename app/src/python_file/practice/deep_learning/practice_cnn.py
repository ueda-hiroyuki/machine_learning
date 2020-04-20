import numpy as np
from sample_data.deep_learning_documents.common.util import *
from python_file.practice.deep_learning.common_layers import *
from python_file.practice.deep_learning.simple_conv_net import *

def main():
    img = np.random.randn(30,1,12,12)
    network = SimpleConvNet()
    out = network.layers['Pool1'].forward(img)



if __name__ == "__main__":
    main()