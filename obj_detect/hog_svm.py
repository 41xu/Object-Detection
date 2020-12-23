import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision.models as models

# hog从图片中提取特征
def get_hog_descriptor(img):
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    h, w = img.shape[:2]
    rate = 64 / w
    img = cv2.resize(img, (64, np.int(rate * h)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    bg = np.zeros((128, 64), dtype=np.uint8)
    bg[:, :] = 127
    h, w = gray.shape
    dy = (128 - h) // 2
    bg[dy:h + dy, :] = gray

    descriptor = hog.compute(bg,winStride=(8,8), padding=(8,8))
    return descriptor

def extract_feature(img):
    vgg16 = models.vgg16(pretrained=True).features[:28]

    vgg16.eval()
    vgg16.cuda()

    img = img.reshape((1, 3, 40, 100))
    img_t = torch.FloatTensor(img).cuda()
    res = vgg16(img_t)
    res_npy = res.data.cpu().numpy()
    return res_npy[0].reshape(1,-1)[0]

def get_data(train_data, labels, path, label_type):
    for filename in os.listdir(path):
        img_dir = os.path.join(path, filename)
        img = cv2.imread(img_dir)
        feature = extract_feature(img)
        train_data.append(feature)
        labels.append(label_type)
        print('{} have done!'.format(filename))
    return train_data, labels


def get_pos_neg_data(p_path, n_path):
    train_data = []
    labels = []
    train_data, labels = get_data(train_data,labels, p_path, 1)

    train_data, labels = get_data(train_data, labels,n_path, -1)
    return np.array(train_data, dtype=np.float32) , np.array(labels, dtype=np.float32)


def svm_train(p_path, n_path):
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    train_data, labels = get_pos_neg_data(p_path, n_path)

    # p_labels, n_labels = np.ones((439,1),dtype=np.int), -np.ones((389,1),dtype=np.int)
    # labels = np.concatenate((p_labels, n_labels))
    resp = np.reshape(labels, [-1, 1])
    # train_data = np.load('train_data.npy')
    print(train_data)
    print(resp)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, resp)
    svm.save('svm_model.dat')

def predict(img_path, model_path):
    img = cv2.imread(img_path)
    img_c = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    svm = cv2.ml.SVM_load(model_path)
    coordinate = []

    for row in range(0, h-40, 5):
        for col in range(0, w-100, 5):
            win_roi = img[row:row+40, col:col+100]
            feat = extract_feature(win_roi)
            feat = np.array(feat, dtype=np.float32).reshape([1,-1])

            print(feat)
            _, result = svm.predict(feat)
            print(result)
            if result[0][0]>0:

                coordinate.append([row, col])
                cv2.rectangle(img_c,(row,col), (row+100,col+40), (0,0,255), 1, 8, 0)
    print(coordinate)
    plt.imshow(img_c)
    plt.show()


if __name__ == '__main__':
    # pre_path = 'CarData/TrainImages/'
    # svm_train(pre_path+'car', pre_path+'other')
    predict('CarData/TestImages/test-0.pgm','svm_model.dat')


