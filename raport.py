import numpy as np
import cv2
import pandas as pd
from scipy.stats.mstats import gmean

def verify_results(img, fov, std):
    i = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    f = cv2.imread(fov, cv2.IMREAD_GRAYSCALE)
    s = cv2.imread(std, cv2.IMREAD_GRAYSCALE)
    TP = FP = TN = FN = 0
    for idx, j in np.ndenumerate(i):
        if f[idx[0], idx[1]] == 0:
            continue
        else:
            if i[idx[0], idx[1]] == 255:
                if i[idx[0], idx[1]] == s[idx[0], idx[1]]:
                    TP += 1
                else:
                    FP += 1
            else:
                if i[idx[0], idx[1]] == s[idx[0], idx[1]]:
                    TN += 1
                else:
                    FN += 1
    return [TP, FP, TN, FN]


img_folder = ['Resized11_h', 'Resized12_h', 'Resized13_h', 'Resized14_h', 'Resized15_h']

for file in img_folder:
    print('Iteration ' + file)
    ip = 'results/' + file + '/' + file + 'Final1.jpg'
    ml = 'mlResults/Result' + file + '.jpg'
    fov = 'Resized/fov/' + file + '_mask.tif'
    std = 'Resized/standard/' + file + '.tif'
    TP1, FP1, TN1, FN1 = verify_results(ip, fov, std)
    TP2, FP2, TN2, FN2 = verify_results(ml, fov, std)
    sen1 = round(TP1/(FN1 + TP1), 2)
    spe1 = round(TN1/(TN1 + FP1), 2)
    pre1 = round(TP1/(TP1 + FP1), 2)
    acc1 = round((TP1 + TN1)/(TP1 + TN1 + FP1 + FN1), 2)
    gmean1 = round(gmean([sen1, spe1]), 2)
    sen2 = round(TP2 / (FN2 + TP2), 2)
    spe2 = round(TN2 / (TN2 + FP2), 2)
    pre2 = round(TP2 / (TP2 + FP2), 2)
    acc2 = round((TP2 + TN2) / (TP2 + TN2 + FP2 + FN2), 2)
    gmean2 = round(gmean([sen2, spe2]), 2)
    l1 = [TP1, FP1, TN1, FN1, sen1, spe1, pre1, acc1, gmean1]
    l2 = [TP2, FP2, TN2, FN2, sen2, spe2, pre2, acc2, gmean2]
    columns = ['TP', 'FP', 'TN', 'FN', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'Geometric mean']
    dict1 = {}
    dict2 = {}
    for i in range(len(l1)):
        dict1[columns[i]] = l1[i]
        dict2[columns[i]] = l2[i]

    print(dict1)
    df = pd.DataFrame(dict1, index=[0])
    df.to_csv('Stats/list1' + file + '.csv')
    df = pd.DataFrame(dict2, index=[0])
    df.to_csv('Stats/list2' + file + '.csv')
