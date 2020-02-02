import cv2
import numpy as np

cap = cv2.VideoCapture(0) # 標準Webカメラ起動(0：内蔵カメラ)
print(cap.isOpened()) # 接続できている場合はTrueが返る
while True:
    is_opened, frame = cap.read() # 接続できている場合はis_openedにTrueが返る
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1) , 0) # ぼかし
    _, thr = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV) # 2値化
    cv2.imshow("OpenCSV Web Camera", thr) # カメラでキャプチャした画像を表示し続ける⇒動画になる
    k =  cv2.waitKey(1) # 1msecのキー入力待ち⇒1msecごとに入力権利がある 
    if k == 27 or k == 13:
        break # escキーかEnterキーが押されるとwhileを抜ける

cap.release() # カメラを閉じる
cv2.destroyWindow() # 現在開いているウィンドウを閉じる