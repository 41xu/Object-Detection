import cv2
import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from obj_detect.nn import Net

def predict(img_path):
    model = Net()
    model.load_state_dict(torch.load('./trained_models/50_model.pth.tar')['state_dict'])
    im = cv2.imread(img_path)
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    maxn = 0
    x,y=0,0
    model.eval()
    for row in range(0, h-40, 5):
        for col in range(0, w-100, 5):
            win_roi = img[row:row + 40, col:col + 100]
            inputs = torch.tensor(win_roi,dtype=torch.float).permute(2,0,1).unsqueeze(0)

            res = model(inputs)[0][0].data
            if res>maxn or res==1:
                print(res)
                x,y = row, col
                maxn = res
                cv2.rectangle(im, (row, col), (row + 100, col + 40), (0, 0, 255), 1, 8, 0)
    print((x,y))
    plt.imshow(im)
    plt.show()

if __name__ == '__main__':

    predict('./CarData/TestImages/test-4.pgm')
