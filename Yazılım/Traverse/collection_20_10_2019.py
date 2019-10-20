import numpy as np
import cv2,time

cam = cv2.VideoCapture(0)

def nothing(x):
    pass

def pointRectColl(point,rect):
    if(point[0] > rect[0][0] and point[0] < rect[1][0] and point[1] > rect[0][1] and point[1] < rect[1][1]):
        return True
    return False

### BASIT DETERMİNANT İLE ALAN HESAPLAMA ###
def approxDeterminant(approx):
    leftCal = approx[len(approx) - 1][0][0] * approx[0][0][1]
    rightCal = approx[len(approx) - 1][0][1] * approx[0][0][0]
    for i in range(len(approx) - 1):
        leftCal += approx[i][0][0] * approx[i + 1][0][1]
        rightCal += approx[i][0][1] * approx[i + 1][0][0]
    result = 0.5 * (abs(leftCal - rightCal))
    return result

### AĞIRLIK MERKEZİ HESAPLAMA ###
def approxCom(approx):
    sumX = 0
    sumY = 0
    for i in range(len(approx)):
        sumX += approx[i][0][0]
        sumY += approx[i][0][1]
    return int(sumX/len(approx)),int(sumY/len(approx))

def com(array):
    sumX = 0
    sumY = 0
    for i in range(len(array)):
        sumX += array[i][0]
        sumY += array[i][1]
    return int(sumX/len(array)),int(sumY/len(array))

img = np.zeros([60, 200, 3], np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('H_up','image',93,255,nothing)
cv2.createTrackbar('S_up','image',255,255,nothing)
cv2.createTrackbar('V_up','image',255,255,nothing)
cv2.createTrackbar('H_lo','image',51,255,nothing)
cv2.createTrackbar('S_lo','image',39,255,nothing)
cv2.createTrackbar('V_lo','image',0,255,nothing)

comArray = []
comArrayCal = (0,0)
last = 0
while(1):
    now = round(time.time())
    _,frame = cam.read()

    #frame = cv2.imread("binary.jpg")
    """
    scale_percent = 20  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Hu = cv2.getTrackbarPos('H_up', 'image') #93
    Su = cv2.getTrackbarPos('S_up', 'image') #255
    Vu = cv2.getTrackbarPos('V_up', 'image') #255
    Hl = cv2.getTrackbarPos('H_lo', 'image') #51
    Sl = cv2.getTrackbarPos('S_lo', 'image') #39
    Vl = cv2.getTrackbarPos('V_lo', 'image') #0

    lower_green = np.array([Hl, Sl, Vl])
    upper_green = np.array([Hu, Su, Vu])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    #gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.medianBlur(gray, 5)
    #thresh1 = cv2.adaptiveThreshold(res, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12)
    contours, h  = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) > 4 and len(approx) < 15:
            cal = approxDeterminant(approx)
            approxCal = approxCom(approx)
            #print("rec")

            if(cal > 50):
                last = round(time.time())
                comArray.append(approxCal)
                comArrayCal = com(comArray)
                cv2.drawContours(frame, [cnt], 0, (25, 25, 255), 2)

                if(len(comArray) > 10):
                    comArray = comArray[1:]
                if(pow(comArrayCal[0] - approxCal[0],2)+pow(comArrayCal[1] - approxCal[1],2) > 500):
                    comArray.clear()

    if(now - last > 3):
        comArrayCal = (0,0)
    rectPoints = [(int(len(frame[0]) * 0.4), int(len(frame) * 0.35)),
                  (int(len(frame[0]) * 0.6), int(len(frame) * 0.65))]
    cv2.rectangle(frame, rectPoints[0],rectPoints[1], ((pointRectColl(comArrayCal,rectPoints))*255, 0, 0), 5)
    cv2.circle(frame, comArrayCal, 3, (255, 255, 255))

    cv2.imshow("Goruntu", res)
    cv2.imshow('image', img)
    cv2.imshow("Contours", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

# or len(approx) == 5 or len(approx) == 6