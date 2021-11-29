import os
import numpy as np
import cv2
from math import ceil
import time
import imutils
from imutils.perspective import four_point_transform
from tensorflow.keras.models import load_model
from os import listdir


def get_x(s):
    return s[1][0]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

def get_x_ver2(s):
    return s[1][0] / s[1][1]


def get_size_box(cnt):
    rect = cv2.boundingRect(cnt)
    return rect[2] * rect[3]


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=2)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=2)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_paper_box(img, cnt):
    boxPoints = order_points(cnt)
    box = four_point_transform(img, boxPoints)

    return box


def get_box(img, cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    docCnt = approx

    if len(docCnt) == 4:
        box = four_point_transform(img, docCnt.reshape(4, 2))
    else:
        rect = cv2.minAreaRect(cnt)
        boxPoints = cv2.boxPoints(rect)
        box = four_point_transform(img, boxPoints)

    return box


def crop_paper(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 17))
    dilation = cv2.dilate(blurred, kernel=kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 25))
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=4)
    canny = cv2.Canny(opening, 5, 100)

    cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=get_size_box, reverse=True)

    paper = get_paper_box(gray, cnts[0])

    return paper


def get_ans_boxes(paper):
    blurred = cv2.GaussianBlur(paper, (5, 5), 1)
    canny = cv2.Canny(blurred, 40, 190)
    cnts = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    xOld, yOld, wOld, hOld = 0, 0, 0, 0
    sorted_ans_blocks = []

    if len(cnts) > 0:
        cnts = sorted(cnts, key=get_x_ver1)
        for c in cnts:
            xCurr, yCurr, wCurr, hCurr = cv2.boundingRect(c)
            if (0.25 > (wCurr / paper.shape[1]) > 0.15) and (0.7 > (hCurr / paper.shape[0]) > 0.5):
                check_xy_min = xCurr * yCurr - xOld * yOld
                check_xy_max = (xCurr + wCurr) * (yCurr + hCurr) - (xOld + wOld) * (yOld + hOld)

                if len(ans_blocks) == 0:
                    ans_blocks.append((get_box(paper, c), [xCurr, yCurr]))
                    xOld = xCurr
                    yOld = yCurr
                    wOld = wCurr
                    hOld = hCurr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append((get_box(paper, c), [xCurr, yCurr]))
                    xOld = xCurr
                    yOld = yCurr
                    wOld = wCurr
                    hOld = hCurr

        sorted_ans_blocks = sorted(ans_blocks, key=get_x)

    return sorted_ans_blocks


def process_ans_blocks(ansBlocks):
    listAnswers = []

    for ansBlock in ansBlocks:
        ansBlockImg = np.array(ansBlock[0])
        offset1 = ceil(ansBlockImg.shape[0] / 6)
        for i in range(6):
            box = np.array(ansBlockImg[i * offset1:(i + 1) * offset1, :])
            blurred = cv2.GaussianBlur(box, (5, 5), 0)
            canny = cv2.Canny(blurred, 10, 60)
            end = 0
            start = 0
            for i in range(canny.shape[0]):
                if np.count_nonzero(canny[i]) > 8:
                    end = i
                else:
                    if end - start > 15:
                        listAnswers.append(box[start:end, :])
                    start = i
    if len(listAnswers) != 120:
        print("extract not enough answers")
        for i, img in enumerate(listAnswers):
            cv2.imwrite("abc/"+str(i)+".jpg", img)

    return listAnswers


def process_list_ans(listAnswers):
    listChoices = []
    for answerImg in listAnswers:
        start = ceil(0.22 * answerImg.shape[1])
        end = answerImg.shape[1] - 15
        bubbleChoice = answerImg[:, start:end]
        bubbleChoice = cv2.resize(bubbleChoice, (112, 112), cv2.INTER_AREA)
        bubbleChoice = bubbleChoice.reshape((112, 112, 1))
        bubbleChoice = bubbleChoice.astype("float32")
        listChoices.append(bubbleChoice)
    return listChoices


def get_id_box(paper):
    blurred = cv2.GaussianBlur(paper, (5, 5), 1)
    imgCanny = cv2.Canny(blurred, 40, 190)
    cnts = cv2.findContours(imgCanny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    xOld, yOld, wOld, h_old = 0, 0, 0, 0

    if len(cnts) > 0:
        cnts = sorted(cnts, key=get_x_ver1)

        for i, c in enumerate(cnts):
            xCurr, yCurr, wCurr, hCurr = cv2.boundingRect(c)
            if (0.2 > (wCurr / paper.shape[1]) > 0.1) & (0.3 > (hCurr / paper.shape[0]) > 0.1):
                check_xy_min = xCurr * yCurr - xOld * yOld
                check_xy_max = (xCurr + wCurr) * (yCurr + hCurr) - (xOld + wOld) * (yOld + h_old)

                if len(ans_blocks) == 0:
                    ans_blocks.append((get_paper_box(paper, c), [xCurr, yCurr]))
                    xOld = xCurr
                    yOld = yCurr
                    wOld = wCurr
                    hOld = hCurr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append((get_paper_box(paper, c), [xCurr, yCurr]))
                    xOld = xCurr
                    yOld = yCurr
                    wOld = wCurr
                    hOld = hCurr

        sorted_ans_blocks = sorted(ans_blocks, key=get_x_ver2, reverse=True)

    return sorted_ans_blocks[0][0]


def process_id_block(id_block):
    listChoices = []

    blurred = cv2.GaussianBlur(id_block, (5, 5), 0)
    canny = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    canny = cv2.bitwise_not(canny)
    otsu = cv2.threshold(id_block, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    end1 = 0
    start1 = 0
    for i in range(canny.shape[1]):
        if np.count_nonzero(canny[:, i]) > 50:
            end1 = i
        else:
            if end1 - start1 > 15:
                listChoices.append([])
                end2 = 0
                start2 = 0
                for j in range(canny.shape[0]):
                    if np.count_nonzero(canny[j]) > 30:
                        end2 = j
                    else:
                        if end2 - start2 > 10:
                            idChoice = otsu[start2:end2, start1:end1]
                            idChoice = cv2.resize(idChoice, (28, 28), cv2.INTER_AREA)
                            idChoice = idChoice.reshape((28, 28, 1))
                            idChoice = idChoice.astype("float32")
                            listChoices[-1].append(idChoice)
                        start2 = j
            start1 = i

    return listChoices


def map_answer(idx):
    if idx == 1:
        answer_circle = 3
    elif idx == 2:
        answer_circle = 2
    elif idx == 4:
        answer_circle = 1
    elif idx == 8:
        answer_circle = 0
    else:
        answer_circle = 4
    return answer_circle


def pre(path, name):
    start = time.time()
    img = cv2.imread(path)
    paper = crop_paper(img)

    # Xu li id
    idBox = get_id_box(paper)
    listIdColumns = process_id_block(idBox)
    idModel = load_model("id_weight.h5")
    ids = []
    for idColumn in listIdColumns:
        idColumn = np.array(idColumn)
        results = idModel.predict_on_batch(idColumn/255.0)
        result = np.argmax(results[:,1])
        ids.append(result)
    print("id la", ids)

    #Xu li cau tra loi
    listAnsBoxes = get_ans_boxes(paper)
    listAns = process_ans_blocks(listAnsBoxes)
    listAns = process_list_ans(listAns)
    listAns = np.array(listAns)

    model = load_model('weight.h5')
    count = 0
    lines = []
    with open("list_ans.txt", "r") as f:
        for line in f:
            lines.append(int(line.rstrip()))

    results = model.predict_on_batch(listAns / 255.0)
    for i, result in enumerate(results):
        a = map_answer(np.argmax(result))
        if a == lines[i]:
            count += 1
        else:
            print(i + 1, a, lines[i], np.argmax(result))
            cv2.imwrite("ex/" + str(name) + "t" + str(i) + ".jpg", listAns[i])

    print("De so", name, "dung", count)
    end = time.time()
    print("chay het", end - start)


# def gen_data(dir, dest):
#     for i, fold in enumerate(os.listdir(dir)):
#         os.mkdir(os.path.join(dest + "/", fold))
#         for j, file in enumerate(os.listdir(dir + "/" + fold)):
#             img = cv2.imread(dir + "/" + fold + "/" + file)
#             paper = crop_paper(img)
#             list_ans_boxes = get_ans_boxes(paper)
#             if len(list_ans_boxes) != 0:
#                 list_ans = process_ans_blocks(list_ans_boxes)
#                 list_ans = process_list_ans(list_ans)
#                 if len(list_ans) != 120:
#                     print(i, j, len(list_ans))
#                 for s, ans in enumerate(list_ans):
#                     cv2.imwrite(dest + "/" + fold + "/" + str(i) + "t" + str(j) + "t" + str(s) + ".jpg", ans)
#
#
# def visualize_data():
#     for folder in listdir("./gendata/"):
#         print(folder, len([file for file in listdir("./gendata/" + folder + "/")]))


if __name__ == '__main__':
    for i, path in enumerate(os.listdir("input6")):
        pre(os.path.join("input6" + "/", path), i)
