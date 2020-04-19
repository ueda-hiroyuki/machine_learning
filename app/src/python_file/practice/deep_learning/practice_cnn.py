import numpy as np
from sample_data.deep_learning_documents.common.util import *
from python_file.practice.deep_learning.common_layers import *

def main():
    img = np.random.randint(0,9,(2,3,4,4))
    pool = Pooling(pool_h=2, pool_w=2)
    out = pool.forward(img)


if __name__ == "__main__":
    main()