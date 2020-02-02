import cv2, os, copy
import matplotlib.pyplot as plt
from sklearn.externals import joblib

clf = joblib.load("../sample_data/fish_model.pkl")
output_dir = "../sample_data/fish_imgs/best_fish"
last_img = None
fish_thr = 1 # 画像を出力するかの閾値
count = 0
frame_count = 0

# ファイルパスにディレクトリが存在しなかったらディレクトリ作成する
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

video = cv2.VideoCapture("../sample_data/book-mlearn-gyomu-master/src/ch3/video/fish.mp4")

while True:
    is_OK, frame = video.read()
    if is_OK is False:
        break

    frame = cv2.resize(frame, (640, 360))
    copy_frame = copy.copy(frame)
    frame_count += 1

    # 画像を2値化する
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (15,15), 0)
    img = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY_INV)[1]

    if not last_img is None:
        diff_img = cv2.absdiff(last_img, img)
        cntrs = cv2.findContours(diff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] # 背景差分の輪郭を抽出

        # 魚の存在を判断する
        fish_count = 0
        for cntr in cntrs:
            x,y,w,h = cv2.boundingRect(cntr)
            if 100 < w < 500:
                # 抽出した領域内に魚がいるのか予測させる
                fish_img = frame[y:y+h, x:x+w] # 画像のトリミング([縦方向の切り取り、横方向の切り取り])
                fish_img = cv2.resize(fish_img, (64, 32))
                img_data = fish_img.reshape(-1,) # トリミングした部分を1次元配列に変換
    
                y_pred = clf.predict([img_data]) # 抽出した領域内に魚が存在するのかを判断する
                
                if y_pred[0] == 1: # frameを1つ1つ判断していくためy_predは長さが1のリスト(0:false,1:true)
                    fish_count += 1 
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        if fish_count > fish_thr:
            print("魚を検出しました!!")
            cv2.imwrite(f'{output_dir}/{count}.jpg', copy_frame) # 魚を検出した画像をファイル出力
            count += 1

    cv2.imshow("diff frame", frame) # 元の画像＋矩形の描画

    k =  cv2.waitKey(1) # 1msecのキー入力待ち⇒1msecごとに入力権利がある 
    if k == 27 or k == 13:
        break # escキーかEnterキーが押されるとwhileを抜ける
    last_img = img

video.release() # カメラを閉じる
cv2.destroyAllWindows() # 現在開いているウィンドウを閉じる
