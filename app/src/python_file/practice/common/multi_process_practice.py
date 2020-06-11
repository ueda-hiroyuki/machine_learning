import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def func():
    time.sleep(1)

def main():
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as e:
        for i in range(8):
            e.submit(func)    
    print (time.time()-start)

if __name__ == "__main__":
    main()