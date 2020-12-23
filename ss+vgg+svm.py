from obj_detect.selective_search import selective_search
import torchvision.models as models
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    img = cv2.imread('CarData/TestImages/test-2.pgm')

    svm = cv2.ml.SVM_load('svm_model.dat')
    img_lbl, regions = selective_search(img, scale=200, sigma=0.8, min_size=400)
    vgg16 = models.vgg16(pretrained=True).features[:28]
    vgg16.eval()
    vgg16.cuda()
    for region in regions[:30]:

        if region['size']<500:
            continue
        left = region['rect'][0]
        top = region['rect'][1]
        w = region['rect'][2]
        h = region['rect'][3]
        if len(img[left:left+w,top:top+h,:]) ==0:
            continue
        img_t = cv2.resize(img[left:left+w,top:top+h,:], (100,40))
        img_t = torch.FloatTensor(img_t).permute(2,0,1).unsqueeze(0).cuda()

        res = vgg16(img_t)
        feat = res.data.cpu().numpy()[0].reshape(1,-1)[0]
        feat = np.array(feat, dtype=np.float32).reshape([1, -1])
        print(feat.shape)

        _, result = svm.predict(feat)
        if result[0][0]>0:
            cv2.rectangle(img, (left, top), (left + w, top + h), (255, 0, 0), 1, 8, 0)
    plt.imshow(img)
    plt.show()