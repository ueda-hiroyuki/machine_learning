import numpy as np
from sample_data.deep_learning_documents.common.util import *
from python_file.practice.deep_learning.common_layers import *
from python_file.practice.deep_learning.simple_conv_net import *

def main():
    img = np.random.randint(0,9,(2,3,4,4))
    network = SimpleConvNet()



if __name__ == "__main__":
    main()