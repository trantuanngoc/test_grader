import os
import imutils
import numpy as np
import cv2
from math import ceil
from imutils.perspective import four_point_transform
from tensorflow.keras.models import load_model


def get_x(s):
    return s[1][0]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def get_size_box(ctn):
    x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(ctn)
    return w_curr * h_curr


def get_box(img, cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    docCnt = approx
    
    if len(docCnt) == 4:
        box = four_point_transform(img, docCnt.reshape(4, 2))
    else:
        rect = cv2.minAreaRect(cnt)
        box_points = cv2.boxPoints(rect)
        box = four_point_transform(img, box_points)
    
    return box


def crop_paper(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    im_bw = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_canny = cv2.Canny(im_bw, 10, 70)
    
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=get_size_box, reverse=True)
    
    paper = get_box(gray_img, cnts[0])
    return paper


def get_ans_boxes(paper):
    blurred = cv2.GaussianBlur(paper, (5, 5), 0)
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
            blurred = cv2.GaussianBlur(box_img, (5, 5), 0)
            canny = cv2.Canny(blurred, 10, 60)
            end = 0
            start = 0
            for i in range(canny.shape[0]):
                if np.count_nonzero(canny[i]) > 7:
                    end = i
                else:
                    if end - start > 15:
                        list_answers.append(box_img[start:end, :])
                    start = i
    
    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    for answer_img in list_answers:
        start = ceil(0.21 * answer_img.shape[1])
        end = answer_img.shape[1] - 15
        bubble_choices = answer_img[:, start:end]
        list_choices.append(bubble_choices)
    return list_choices

def create_dataset(dir):
    for i, path in enumerate(os.listdir(dir)):
        img = cv2.imread(os.path.join("dataset/", path))
        paper = crop_paper(img)
        list_ans_boxes = get_ans_boxes(paper)
        list_ans = process_ans_blocks(list_ans_boxes)
        list_ans = process_list_ans(list_ans)
        for j, ans in enumerate(list_ans):
            cv2.imwrite("data/" + str(i) + "t" + str(j) + ".jpg", ans)
            
def map_answer(idx):
    if idx == 0:
        answer_circle = "A"
    elif idx == 1:
        answer_circle = "B"
    elif idx == 2:
        answer_circle = "No choice"
    elif idx == 3:
        answer_circle = "C"
    elif idx == 4:
        answer_circle = "D"
    return answer_circle

if __name__ == '__main__':
    img = cv2.imread("test/z2914711115171_40fd4a730c6601d420fb9ead93074ae1.jpg")
    paper = crop_paper(img)
    list_ans_boxes = get_ans_boxes(paper)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    model = load_model('weight.h5')
    for i, ans in enumerate(list_ans):
        img = cv2.resize(ans, (28, 28), cv2.INTER_AREA)
        img = img.reshape((28, 28, 1))
        img = np.expand_dims(img, axis=0)
        print(i+1, map_answer(np.argmax(model.predict(img))))

