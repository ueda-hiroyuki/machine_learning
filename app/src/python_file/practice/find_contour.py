import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../sample_data/flower.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.GaussianBlur(gray_img, (1, 1), 0) #画像の平滑化(細かすぎる部分を検出しないようにするため)
img2 = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)[1] # 画像の二値化(第二引数を超えた場合、第三引数の値になる(50を超えたものはすべて200の色になる))

plt.subplot(1, 2, 1)
plt.imshow(img2, cmap="gray")

cntrs = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] # 輪郭を抽出(引数によって抽出する値が異なる)
for cntr in cntrs:
    x,y,w,h = cv2.boundingRect(cntr) # 矩形を抽出
    if w < 50 or w > 200: continue # ノイズを除去(大きすぎるもの、小さすぎるもの)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2) # rectangle(矩形の頂点１、矩形の頂点1の反対側の頂点、矩形の色、矩形の太さ)


plt.subplot(1, 2, 2) # 一画像の中にグラフ(今回は画像)を2つ配置する
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # imshow: 貼り付ける
plt.show() # show: 表示する