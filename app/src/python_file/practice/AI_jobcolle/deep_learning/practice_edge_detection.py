import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

SAMPLE_DIR = "src/sample_data/AI_jobcolle"
IMG_PATH = f"{SAMPLE_DIR}/peppers.png"

FILTERS = [
        # 縦線を抽出するフィルタ
        [
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]
        ],
        # 横線を抽出するフィルタ
        [
            [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]],
            [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]],
        
            [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]
        ],
        # 赤い部分だけ抽出するフィルタ
        [
            [[0, 0, 0],
            [0, 10, 0],
            [0, 0, 0]],
            [[0, 0, 0],
            [0, -10, 0],
            [0, 0, 0]],
            [[0, 0, 0],
            [0, -10, 0],
            [0, 0, 0]],
        ],
        # 緑の部分の輪郭を抽出するフィルタ
        [
            [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
            [[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]],
            [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
        ],
        # ぼかすフィルタ(平均化フィルタ)
        [
            [[1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]],
            [[1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]],
            [[1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]],
        ],
    ]

def im2col(input_data, filter_size, stride=1, pad=0):
    N, C, H, W = input_data.shape
    filter_h, filter_w = filter_size[0], filter_size[1]

    out_h = (H + 2*pad - filter_h)//stride + 1 # 出力される画像の高さ
    out_w = (W + 2*pad - filter_w)//stride + 1 # 出力される画像の幅

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') # zero padding
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def main():
    
    filters = np.asarray(FILTERS)

    original_image = Image.open(IMG_PATH)
    # 入力した画像をnarray変換し、チャネル数を0列目に移動
    num_img = np.array(original_image).transpose(2,0,1) 

    N = 1 # バッチ数(今回は1枚)
    C, W, H = num_img.shape
    num_img = num_img.reshape(N, C, W, H) # (1, 3, 512, 512)に変換

    FN = filters.shape[0] # 3×3の3チャンネルフィルターが5つ
    PAD = 0
    S = 1
    FW = 3
    FH = 3

    col = im2col(num_img, (FW, FH)) # 畳み込み前に画像データを成形する

    out_h = (H + 2*PAD - FH)//S + 1
    out_w = (W + 2*PAD - FW)//S + 1

    f = filters.reshape(FN, -1).T

    result = np.dot(col, f) # 畳み込みを行列計算で行う
    result = result.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)[0] # 各フィルタにかけられた出力画像が5枚(フィルタ数と同じ)
    
    plt.figure(figsize=(15,10))
    plt.gray()

    plt.subplot('231')
    plt.imshow(original_image)
    plt.subplot('232')
    plt.imshow(result[0])
    plt.subplot('233')
    plt.imshow(result[1])
    plt.subplot('234')
    plt.imshow(result[2])
    plt.subplot('235')
    plt.imshow(result[3])
    plt.subplot('236')
    plt.imshow(result[4])
    
    plt.savefig(f"{SAMPLE_DIR}/output_figures.png")




if __name__ == "__main__":
    main() 