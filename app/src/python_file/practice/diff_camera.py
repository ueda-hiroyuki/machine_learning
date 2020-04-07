import cv2

cap = cv2.VideoCapture(0)
last_img = None

while True:
    frame = cap.read()[1]
    frame = cv2.resize(frame, (700, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9) , 0) 
    _, thr = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV) 
    
    if last_img is None:
        last_img =  thr
        continue
    diff_frame = cv2.absdiff(last_img, thr) # 2つの画像の差分を出力(背景差分)
    cntrs = cv2.findContours(diff_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] # 差分のある部分の輪郭を出力
    
    for cntr in cntrs:
        x,y,w,h = cv2.boundingRect(cntr) # 輪郭部分の座標を抽出
        if w > 100: # 小さい変化(ノイズ)は除去
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2) # 矩形を描画
    
    cv2.imshow("diff frame", frame) # 元の画像＋矩形の描画
    cv2.imshow("mono", thr) # 元の画像＋矩形の描画

    k =  cv2.waitKey(1) # 1msecのキー入力待ち⇒1msecごとに入力権利がある 
    if k == 27 or k == 13:
        break # escキーかEnterキーが押されるとwhileを抜ける

cap.release() # カメラを閉じる
cv2.destroyAllWindows() # 現在開いているウィンドウを閉じる