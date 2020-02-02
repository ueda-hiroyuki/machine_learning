import cv2
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def detect_number(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    img = img[0:h//4, w//2:]

    # 画像を二値化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (1,1), 0)
    img2 = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
    
    plt.subplot(2, 1, 1)
    plt.imshow(img2, cmap="gray")

    # 輪郭を抽出
    cntrs = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    result = []
    for cntr in cntrs:
        x, y, w, h = cv2.boundingRect(cntr)
        if (10 < w < 20) and (h > 20):  #　大きすぎる小さすぎるものを排除
            result.append([x, y, w, h])

    result = sorted(result, key=lambda x: x[0])

    result2 = []
    l = 1
    for x, y, w, h in result:
        if (x - l) > 3:
            result2.append([x, y, w, h])
            l = x

    for x, y, w, h in result2:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
    print(result2)
    return result2, img 

if __name__ == "__main__":
    result, img = detect_number("../sample_data/postcard.jpg")
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

