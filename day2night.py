import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from PIL import Image


def get_mean_and_std(img):
    x_mean, x_std = cv2.meanStdDev(img)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

def op_each_channel(source_lab, target_lab):

    L_s, A_s, B_s = cv2.split(source_lab)
    L_t, A_t, B_t = cv2.split(target_lab)
    #### histogram matching
    A_new = style_transfer(A_s, A_t)
    B_new = style_transfer(B_s, B_t)
    # A_new = cv2.convertScaleAbs(A_new)
    # B_new = cv2.convertScaleAbs(B_new)
    gamma = 2.2
    L_new = np.power(L_s/255.0, gamma) ## gamma correction
    L_new = L_new*255
    img_n = cv2.merge([L_new, A_new, B_new])
    return img_n

def color_transfer(source, target):

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab =  cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    # s_mean, s_std = get_mean_and_std(source_lab)
    # t_mean, t_std = get_mean_and_std(target_lab)
    # img_n = ((source_lab-s_mean)/s_std)*t_std+t_mean
    img_n = op_each_channel(source_lab, target_lab)
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    img_n = cv2.convertScaleAbs(img_n)
    dst = cv2.cvtColor(img_n, cv2.COLOR_LAB2BGR)
    return dst

def equalize_hist_color(img):

    channels = cv2.split(img) ##BGR
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch)) # each channel cv2.equalizeHist()
    eq_image = cv2.merge(eq_channels)
    return eq_image

def image_hist(img, title):
    # plt.hist(img.ravel(), 30, [0, 256])
    color = ('b', 'g', 'r')
    for id, bgrcolor in enumerate(color):
        hist = cv2.calcHist([img], [id], None, [256], [0.0, 255.0])
        plt.plot(hist, color=bgrcolor)
        plt.title('Histrogram of ' + title + ' Color image')
    plt.show()

def style_transfer(image, ref, multichannel=False): ## a,b channel
    matched = match_histograms(image, ref, multichannel=multichannel)
    return matched


if __name__ == '__main__':


    img11 = cv2.imread("/disks/disk0/feifei/data/semantic_data/Cityscapes/leftImg8bit/train/erfurt/erfurt_000001_000019_leftImg8bit.png") ## day
    img21 = cv2.imread("/disks/disk0/feifei/data/semantic_data/Zurich/train/rgb_anon/train/night/GOPR0351/GOPR0351_frame_000090_rgb_anon.png") #night
    img1 = Image.open("/disks/disk0/feifei/data/semantic_data/Cityscapes/leftImg8bit/train/zurich/zurich_000055_000019_leftImg8bit.png").convert('RGB')
    img2 = Image.open("/disks/disk0/feifei/data/semantic_data/Zurich/train/rgb_anon/train/night/GP010376/GP010376_frame_000191_rgb_anon.png").convert('RGB')

    fig, axes = plt.subplots(1,2, figsize=(8,8))
    for i, img in enumerate((img11[:,:,::-1], img21[:,:,::-1])):
        axes[i].imshow(img)
    plt.tight_layout()
    plt.show()

    ## BGR color Histrogram
    img1 = np.asarray(img1)[:,:,::-1]
    img2 = np.asarray(img2)[:, :, ::-1]
    out = color_transfer(img1, img2)
    image_hist(img1, 'Day')
    image_hist(img2, 'Night')
    image_hist(out, 'Transfer')

    ## Show
    # cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Demo", 800, 600)
    # cv2.imshow("Demo", out)
    # key = cv2.waitKey(0)
    # if key  == ord('q'):
    #     cv2.destroyAllWindows()
    cv2.imwrite("out.jpg", out)


