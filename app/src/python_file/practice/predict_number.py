import cv2
import joblib

def predict(file_name):
    clf = joblib.load("../sample_data/digits.pkl")
    
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (8, 8))
    img = 15 - img
    img = img.reshape((-1, 64))
    
    pred = clf.predict(img)
    return pred[0]

res = predict("../sample_data/5.png")
print(res)