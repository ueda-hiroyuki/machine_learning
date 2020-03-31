import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(x*w) 
    if tmp == 1:
        return 1
    else:
        return 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(x*w) 
    if tmp > 0:
        return 1
    else:
        return 0


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(x*w) 
    if tmp == 1:
        return 0
    else:
        return 1

def XOR(x1, x2):
    res1 = NAND(x1,x2)
    res2 = OR(x1,x2)
    res = AND(res1, res2)
    return res


if __name__ == "__main__":
    res = XOR(1,0)    
    print(res)