import cv2
import numpy as np
from copy import deepcopy

#**************************************************#
#--------------------彩色变换----------------------#
#1.转换为灰度图
def rgb2gray(img):
    try:
        img = deepcopy(img)
        rgb = img[:,:,0:3]
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        img[:,:,0:3] = np.stack((gray,gray,gray),axis=2)
        return img
    except:
        return ValueError

#2.伪彩色
def myPseudoColor(img, colormap=cv2.COLORMAP_JET):
    '''colormap可以选择cv2.COLORMAP_RAINBOW，cv2.COLORMAP_JET，cv2.COLORMAP_HSV等'''
    try:
        img = deepcopy(img)
        rgb = img[:,:,0:3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        dst = cv2.applyColorMap(gray, colormap)
        img[:,:,0:3] = dst
        return img
    except:
        return ValueError
#**************************************************#
#--------------------正交变换----------------------#
#1.DFT
def dft_one(img):
    try:
        img = deepcopy(img)
        rgb = img[:, :, 0:3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # 进行DFT变换
        f = np.fft.fft2(gray)
        # 将零频点移到频谱的中心
        fshift = np.fft.fftshift(f)
        # 频谱范围很大，故用对数变换来改善视觉效果。加1避免出现log0。
        mag = 20 * np.log(1 + np.abs(fshift))
        #img[:, :, 0:3] = np.stack((mag,mag,mag),axis=2)
        img = mag
        return img
    except:
        return ValueError

#2.DCT
def dct_one(img):
    try:
        img = deepcopy(img)
        rgb = img[:, :, 0:3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # 进行离散余弦变换
        img_dct = cv2.dct(gray)
        # 因为傅立叶频谱范围很大，所以用log对数变换来改善视觉效果。加1避免出现log0。
        mag = 20 * np.log(1 + np.abs(img_dct))
        #img[:, :, 0:3] = np.stack((mag,mag,mag),axis=2)
        img = mag
        return img
    except:
        return ValueError

#**************************************************#
#--------------------空间滤波----------------------#
#1.平滑---------------------------------------------
def myBlur(img, ksize=(5,5)):
    '''均值滤波'''
    try:
        img = deepcopy(img)
        return cv2.blur(img, ksize)
    except:
        return ValueError


def myMedianBlur(img, ksize=5):
    '''中值滤波，适合椒盐噪声'''
    try:
        img = deepcopy(img)
        return cv2.medianBlur(img, ksize)
    except:
        return ValueError


def myGaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0):
    '''高斯滤波，适合随机噪声'''
    try:
        img = deepcopy(img)
        return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
    except:
        return ValueError


def myBilateralFilter(img, d=30, sigmaColor=50, sigmaSpace=50):
    '''双边滤波（Bilateral filter）是一种非线性的滤波方法，是结合图像的空间邻近度
    和像素值相似度的一种折中处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的。
    d为邻域范围，sigmaColor为颜色滤波范围, sigmaSpace为空间滤波范围'''
    try:
        img = deepcopy(img)
        rgb = img[:, :, 0:3]
        img[:, :, 0:3] = cv2.bilateralFilter(rgb, d, sigmaColor, sigmaSpace)
        return img
    except:
        return ValueError

def myNlMeans(img, h=15, hColor=15):
    '''非局部均值去噪,适合高斯噪声'''
    try:
        img = deepcopy(img)
        rgb = img[:, :, 0:3]
        img[:, :, 0:3] = cv2.fastNlMeansDenoisingColored(rgb, None, h, hColor, 7, 21)
        return img
    except:
        return ValueError
#2.锐化--------------------------------------------
def mySobel(img, dx=1, dy=0, ksize=3):
    try:
        img = deepcopy(img)
        # dx=1,dy=0时对x方向求导；反之对y方向求导
        edge = cv2.Sobel(img, -1, dx, dy,ksize)
        sharpen = img + edge
        return sharpen
    except:
        return ValueError

def myScharr(img, dx=1, dy=0, ksize=3):
    try:
        img = deepcopy(img)
        # dx=1,dy=0时对x方向求导；反之对y方向求导
        edge = cv2.Sobel(img, -1, dx, dy, ksize)
        sharpen = img + edge
        return sharpen
    except:
        return ValueError

def myLaplacian(img, ksize=3):
    try:
        img = deepcopy(img)
        edge = cv2.Laplacian(img, -1, None, ksize)
        sharpen = img - edge
        return sharpen
    except:
        return ValueError
#******************************************************#
#--------------------直方图均衡化----------------------#
def myHistEqual(img):
    try:
        img = deepcopy(img)
        rgb = img[:, :, 0:3]
        img_hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        # equalize the histogram of the Y channel
        V = img_hsv[:, :, 2]
        V = cv2.equalizeHist(V)
        img_hsv[:, :, 2] = V
        # convert the YUV image back to RGB format
        img[:,:,0:3] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img
    except:
        return ValueError
        
#*************************************************#
#---------------------形态学----------------------#
def myDilate(img, ksize=(5,5)):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        dst = cv2.dilate(img, kernel)
        return dst
    except:
        return ValueError

def myErode(img, ksize=(5,5)):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        dst = cv2.erode(img, kernel)
        return dst
    except:
        return ValueError

def myGradiant(img):
    try:
        img = deepcopy(img)
        rgb = img[:,:,0:3]
        img[:,:,0:3]  = myDilate(rgb) - myErode(rgb)
        return img
    except:
        return ValueError
        
#***************************************************#
#---------------------图像分割----------------------#
def myCanny(img):
    try:
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 获取灰度图
        img = deepcopy(img)
        rgb = img[:, :, 0:3]
        edges = cv2.Canny(rgb, 50, 200, apertureSize=3)  # 边缘检测
        img[:, :, 0:3] = np.stack((edges,edges,edges),axis=2)
        return img
    except:
        return ValueError

def myHoughLines(img, threshold=200):
    try:
        img = deepcopy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 获取灰度图
        edges = cv2.Canny(gray, 50, 200, apertureSize = 3)  # 边缘检测
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # 霍夫变换返回的就是极坐标系中的两个参数  rho和theta

        lines = lines[:, 0, :]  # 将数据转换到二维
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # 由参数空间向实际坐标点转换
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0, 255))
        return img
    except:
        return ValueError

def myHoughCircles(img):
    try:
        img = deepcopy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=30, minRadius=10, maxRadius=50)
        if circles is None:
            exit(-1)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0, 255), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255, 255), 3)
        return img
    except:
        return ValueError

def myWaterShed(img):
    try:
        img = deepcopy(img)
        rgb = img[:,:,0:3]
        # Step1. 加载图像
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # Step2.阈值分割，将图像分为黑白两部分
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Step3. 对图像进行“开运算”，先腐蚀再膨胀
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Step5.通过distanceTransform获取前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
        # Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域
        sure_fg = np.uint8(sure_fg)
        unknow = cv2.subtract(sure_bg, sure_fg)
        # Step7. 连通区域处理
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknow == 255] = 0
        # Step8.分水岭算法
        markers = cv2.watershed(rgb, markers)
        img[markers == -1] = [255, 0, 0, 255]  # 标记分水岭
        return img
    except:
        return ValueError

def myGrabCut(img, iterCount=10):
    try:
        img = img.copy()
        rgb = img[:,:,0:3]
        mask = np.zeros(img.shape[:2], np.uint8)
        SIZE = (1, 65)
        bgdModle = np.zeros(SIZE, np.float64)
        fgdModle = np.zeros(SIZE, np.float64)
        rect = (1, 1, img.shape[1], img.shape[0])
        cv2.grabCut(rgb, mask, rect, bgdModle, fgdModle, iterCount, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img *= mask2[:, :, np.newaxis]
        return img
    except:
        return ValueError

  
#**************************************************#
#---------------------阈值处理----------------------#
def myOSTU(img):
    try:
        img = deepcopy(img)
        rgb = img[:,:,0:3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img[:, :, 0:3] = np.stack((dst,dst,dst),axis=2)
        return img
    except:
        return ValueError

def myAdaptiveThreshold(img, blksize=19):
    try:
        img = deepcopy(img)
        rgb = img[:,:,0:3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, blksize, 5)
        img[:, :, 0:3] = np.stack((dst,dst,dst),axis=2)
        return img
    except:
        return ValueError


#************************************************#
#---------------------去水印----------------------#
def myInpaint(img):
    img = deepcopy(img)
    rgb = img[:,:,0:3]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 通过阈值化获取mask
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # 对mask进行膨胀处理，确保完全覆盖水印区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel)
    # 去水印
    dst = cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)
    img[:,:,0:3] = dst
    #cv2.imshow('gray',mask)
    return img

