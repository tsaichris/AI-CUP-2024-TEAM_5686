import cv2
import glob
import numpy as np
import random
import os

def edge_detection(img,ground_truth=None, show = False):
    img_blur = cv2.GaussianBlur(img, (5, 5), 3)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # Edge detection method
    laplacian = cv2.Laplacian(img_gray,-1 , ksize = 3) #img, ddepth, ksize, scale
    sobel = cv2.Sobel(img_gray, -1, 1, 1, 1, 7) #img, ddepth, dx, dy, ksize, scale
    canny = cv2.Canny(img_gray, 36, 36) #img, threshold1, threshold2
    
    # Thresholding to get a binary image
    _, laplacian_binary = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, sobel_binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Union of Laplacian, Sobel, and Canny
    union = cv2.bitwise_or(laplacian_binary, canny)
    #union = cv2.bitwise_or(union, sobel_binary)
    # Intersection of Laplacian, Canny, and Sobel
    intersection = cv2.bitwise_and(laplacian_binary, canny)
    #intersection = cv2.bitwise_and(intersection, sobel_binary)

    if show:
        cv2.imshow('original',img)
        cv2.imshow('GT',ground_truth)
        cv2.imshow('blur+gray',img_gray)
        cv2.imshow('lap_binay',laplacian_binary)
        cv2.imshow('sobel',sobel)
        cv2.imshow('canny',canny)
        cv2.imshow('Intersection', intersection)
        cv2.imshow('Union', union)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    return union,intersection


if __name__ == "__main__":
    images_folder = 'C://Users//User//Desktop//drone//Training_dataset//img'
    labels_folder = 'C://Users//User//Desktop//drone//Training_dataset//label_img_converted'

    images_names = sorted(glob.glob(f"{images_folder}/*"))
    # Randomly select image file names
    selected_images_names = random.sample(images_names, 10)


    for imname in selected_images_names:
        img = cv2.imread(imname, 1)
        if img is not None:
            # Get the corresponding ground truth mask name
            img_name = os.path.basename(imname)
            label_name = os.path.join(labels_folder, img_name)
            
            # Load the ground truth mask
            ground_truth = cv2.imread(label_name, 0)
        if ground_truth is not None:
            union,intersection = edge_detection(img,ground_truth = ground_truth, show = True)
        

