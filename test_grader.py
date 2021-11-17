import os
import imutils
import numpy as np
import cv2
from math import ceil
from imutils.perspective import four_point_transform


def get_x(s):
    return s[1][0]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def get_size_box(ctn):
    x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(ctn)
    return w_curr * h_curr


def get_box(img, cnt):
    D = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * D, True)
    docCnt = approx
    
    if len(docCnt) == 4:
        paper = four_point_transform(img, docCnt.reshape(4, 2))
    else:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        paper = four_point_transform(img, box)
    
    return paper


def crop_paper(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    (thresh, im_bw) = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_canny = cv2.Canny(im_bw, 10, 70)
    
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=get_size_box, reverse=True)
    
    D = cv2.arcLength(cnts[0], True)
    approx = cv2.approxPolyDP(cnts[0], 0.04 * D, True)
    docCnt = approx
    
    if len(docCnt) == 4:
        paper = four_point_transform(img, docCnt.reshape(4, 2))
    else:
        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)
        paper = four_point_transform(img, box)
    
    return paper


def get_ans_boxes(img):
    paper = crop_paper(img)
    gray_img = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0
    
    if len(cnts) > 0:
        cnts = sorted(cnts, key=get_x_ver1)
        
        for c in cnts:
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            if 0.19 > (w_curr * h_curr) / (paper.shape[0] * paper.shape[1]) > 0.1:
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)
                
                if len(ans_blocks) == 0:
                    ans_blocks.append((get_box(paper, c), [x_curr, y_curr]))
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append((get_box(paper, c), [x_curr, y_curr]))
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
        
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        
    return sorted_ans_blocks


def process_ans_blocks(ans_blocks):
    list_answers = []

    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])
        offset1 = ceil(ans_block_img.shape[0] / 6)
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.Canny(blurred, 10, 60)
            
            end = 0
            start = 0
            for i in range(thresh.shape[0]):
                if np.count_nonzero(thresh[i]) > 7:
                    end = i
                else:
                    if end - start > 15:
                        list_answers.append(box_img[start:end, :])
                    start = i

    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    for i, answer_img in enumerate(list_answers):
        start = ceil(0.2 * answer_img.shape[1])
        end = answer_img.shape[1] - 15
        bubble_choice = answer_img[:, start:end]
        list_choices.append(bubble_choice)
    return list_choices


if __name__ == '__main__':
    for i, path in enumerate(os.listdir("dataset")):
        img = cv2.imread(os.path.join("dataset/", path))
        list_ans_boxes = get_ans_boxes(img)
        list_ans = process_ans_blocks(list_ans_boxes)
        list_ans = process_list_ans(list_ans)
        for j, ans in enumerate(list_ans):
            cv2.imwrite("data/" + str(i) + "t" + str(j) + ".jpg", ans)
    
