import cv2
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from detect_postcard_number import detect_number

# 学習済みの手書き数字データを読み込む
clf = joblib.load("../sample_data/digits.pkl")

# 輪郭領域を読み取る。
result, img = detect_number("../sample_data/postcard.jpg")

for i, cntr in enumerate(result):
    x, y, w, h = cntr
    x += 3
    y += 4
    w -= 6
    h -= 6
    img2 = img[y:y+h, x:x+w]

    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.resize(img2_gray, (8,8))
    img2_gray = 15 - img2_gray // 16
    img2_gray = img2_gray.reshape((-1, 64))
    res = clf.predict(img2_gray)

    plt.subplot(1, 11, i+1)
    plt.imshow(img2)
    plt.axis("off")
    plt.title(res)

plt.show()