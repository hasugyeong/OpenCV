import numpy as np
import cv2

imageFile = './data/paper.jpg'   # 영상 파일 이름

img = cv2.imread(imageFile) # cv2.imread(imageFile, cv2.IMREAD_COLOR)

height, width, channel = img.shape



src1 = cv2.imread('./data/duksung_symbol2.png')
src2 = cv2.resize(src1, dsize=(100, 100))

list=[]
start=0

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        
         cv2.circle(param[0], (x, y), 5, (255, 0, 0), 3)
         list.append([x,y])
       
         global start
         start=start+1
         print(list,start)
         if start==4:
             srcPoint=np.array([list[0],list[1],list[2],list[3]], dtype=np.float32)
             dstPoint=np.array([[0, 0],[0,height] , [width, height], [width, 0]], dtype=np.float32)
             matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
             img_perspective = cv2.warpPerspective(img, matrix, (width, height))
             #cv2.imshow('Perspective', img_perspective)
             
             rows,cols,channels = src2.shape
             roi = img_perspective[0:rows, 0:cols]
             gray = cv2.cvtColor(src2,cv2.COLOR_BGR2GRAY)
             ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
             mask_inv = cv2.bitwise_not(mask)
             src1_bg = cv2.bitwise_and(roi, roi, mask = mask)
             src2_fg = cv2.bitwise_and(src2, src2, mask = mask_inv)
             dst = cv2.bitwise_or(src1_bg, src2_fg)
             img_perspective[0:rows, 0:cols] = dst
             cv2.imshow('result',img_perspective)

           
    cv2.imshow("img", param[0])

cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse, [img])





cv2.waitKey()
cv2.destroyAllWindows()
