import math
import cv2
import csv
from collections import Counter
import copy
import numpy as np
import random
import pandas as pd
import time

src4=cv2.imread("sejong_univ.png")
dst4=src4.copy()

src = cv2.imread("only_purple_map1.png")  # 원본 이미지

cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()

dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cv2.drawContours(dst, [contours[i]], 0, (0, 0, 0), 2)
    # cv2.putText(dst, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
    print(i, hierarchy[0][i])

    # cv2.waitKey(0)

cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()

src2 = cv2.imread("purple_map.png")  # 원본 이미지
cv2.imshow('src2',src2)
cv2.waitKey()
cv2.destroyAllWindows()

dst2 = src2.copy()
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, 5, param1=25, param2=10, minRadius=7, maxRadius=10)
circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=21, minRadius=13, maxRadius=17)
print(circles[0])
print(circles2[0])
for i in circles[0]:
    cv2.circle(dst2, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)

for i in circles2[0]:
    cv2.circle(dst2, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)

for i in range(len(contours)):
    cv2.drawContours(dst2, [contours[i]], 0, (0, 0, 0), 2)
    # cv2.putText(dst2, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
    print(i, hierarchy[0][i])

    # cv2.waitKey(0)
cv2.imshow('dst2',dst2)
cv2.waitKey()
cv2.destroyAllWindows()


all_circle = []

for i in range(len(circles[0])):
  tmp = [int(circles[0][i][0]),int(circles[0][i][1])]
  all_circle.append(tmp)
for i in range(len(circles2[0])):
  tmp = [int(circles2[0][i][0]),int(circles2[0][i][1])]
  all_circle.append(tmp)

#print(circles[0])
#print(len(circles2[0]))
print(all_circle)

for k in range(len(contours)):
    # print('______________________')
    for m in range(len(contours[k])):
        x = contours[k][m][0][0]
        y = contours[k][m][0][1]
        # print(x , y)

def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result


#original 출발점, connected 도착점 weight 사이 거리값

original = []
connected = []
weight = []



for i in range(len(all_circle)):
  for j in range(i+1,len(all_circle)):
    dst3 = src.copy()
    dst3 = cv2.line(dst3, (all_circle[i][0],all_circle[i][1]), (all_circle[j][0],all_circle[j][1]), (100,200,255), 2, 8, 0)
    gray3 = cv2.cvtColor(dst3, cv2.COLOR_BGR2GRAY)
    ret3, binary3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)
    binary3 = cv2.bitwise_not(binary3)
    contours3, hierarchy3 = cv2.findContours(binary3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if(len(contours3)>len(contours)):
      continue
    else:
      dst2 = cv2.line(dst2, (all_circle[i][0],all_circle[i][1]), (all_circle[j][0],all_circle[j][1]), (100,200,255), 1, 8, 0)
      # csv를 만들기 위한 리스트 구현
      ori = int(i)
      con = int(j)
      wei = distance(all_circle[i][0],all_circle[i][1],all_circle[j][0],all_circle[j][1])
      original.append(ori)
      connected.append(con)
      weight.append(wei)

cv2.imshow('ds2_2',dst2)
cv2.waitKey()
cv2.destroyAllWindows()

tmp_pdf = {
    'original':original,
    'connected':connected,
    'weight':weight
}

pdf = pd.DataFrame(tmp_pdf)

print(pdf)

#pdf --> csv
# 구글 드라이브 내 AI-SW-HACK 폴더에 gragh라는 이름으로 csv 저장
pdf.to_csv('graph.csv',header=True, index=False)
