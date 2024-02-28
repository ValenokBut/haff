import cv2
import skimage
import numpy as np
from scipy.ndimage.filters import sobel
import matplotlib.pyplot as plt
from collections import defaultdict

def calc_grad(pat_im):
    #получение границ паттерного изображения
    borders = skimage.feature.canny(pat_im, low_threshold=10, high_threshold=70)
    
    #рассчет градиетов для паттерного изображения
    gradient = np.arctan2(sobel(borders, axis=0, mode='constant'),
                          sobel(borders, axis=1, mode='constant')) * (180 / np.pi)
    
    return gradient, borders

def calc_R_table(gradient_patt, borders_patt, c):
    # получение R_table
    R_table = defaultdict(list)
    for (x, y), value in np.ndenumerate(borders_patt):
        if value: R_table[gradient_patt[x,y]].append((c[0]-x, c[1]-y))
    
    return R_table

def hough_transform(pat_im, test_img):
    pat_im = cv2.imread(pat_im, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
       
    #центр паттерного изображения
    c = (pat_im.shape[0]/2, pat_im.shape[1]/2)
    
    #получение градиетов и границ для паттерного и тестового изображений
    gradient_patt, borders_patt = calc_grad(pat_im)
    gradient_test, borders_test = calc_grad(test_img)
    
    # получение R_table
    R_table = calc_R_table(gradient_patt, borders_patt, c)
    
    #угол
    sk = 1
    
    # проведение голосования
    voit_list = np.zeros((test_img.shape[0], test_img.shape[1], int(360/sk)))
    for (x, y), value in np.ndenumerate(borders_test):
        if value:
            for pair in R_table[gradient_test[x,y]]:
                xs, ys = pair[0], pair[1]
                for k in range(voit_list.shape[2]):
                    xc = x + (xs * np.cos(k * sk * np.pi / 180) - ys * np.sin(k * sk * np.pi / 180))
                    yc = y + (xs * np.sin(k * sk * np.pi / 180) + ys * np.cos(k * sk * np.pi / 180))
                    if xc < voit_list.shape[0] and yc < voit_list.shape[1]:
                        voit_list[int(xc)][int(yc)][int(k)] += 1
    
    # картинка для визуализации голования
    visual_voit = np.zeros((voit_list.shape[0], voit_list.shape[1]))
    for i in range(voit_list.shape[0]):
        for j in range(voit_list.shape[1]):
            visual_voit[i][j] = np.sum(voit_list[i][j][:])
            
    plt.figure()
    plt.title('visualization_of_voiting')
    plt.imshow(visual_voit)
    plt.savefig('visualization_of_voiting.png')
    plt.close()
    
    plt.figure()
    plt.title('visualization_of_detecting')
    plt.imshow(test_img)
    amount_af_points = 20
    flattened_arr = [item for sublist in visual_voit for item in sublist]
    indices = sorted(range(len(flattened_arr)), key=lambda x: flattened_arr[x], reverse=True)[:amount_af_points]
    row_length = len(visual_voit[0])
    result_indices = [(i // row_length, i % row_length) for i in indices]
    x_points = [t[1] for t in result_indices]
    y_points = [t[0] for t in result_indices] 
    plt.scatter(x_points, y_points, marker='o', color='g')
    plt.savefig('visualization_of_detecting.png')
    plt.close()
    return voit_list
    
if __name__ == '__main__':
    hough_transform(pat_im = "l.png", test_img = "l_test.png")