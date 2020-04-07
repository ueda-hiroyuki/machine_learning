import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

for num in range(15):
    plt.subplot(3, 5, num + 1)
    plt.axis("off")
    plt.title(digits.target[num])
    plt.imshow(digits.images[num])

plt.show()