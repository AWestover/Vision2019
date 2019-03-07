import cv2
import numpy as np
import time
from networktables import NetworkTables as nt


frameScalingFactor = 0.3

# PARAMS
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100000#*frameScalingFactor*frameScalingFactor
params.maxArea = 1000000#*frameScalingFactor*frameScalingFactor
params.filterByCircularity = False
params.filterByColor = False
params.blobColor = 255
params.filterByConvexity = False
params.minConvexity = 0.4
params.maxConvexity = 1.0
params.filterByInertia = False
params.minInertiaRatio = 0.0
params.maxInertiaRatio = 0.4
detector = cv2.SimpleBlobDetector_create(params)

LBOUND_ORANGE = np.array([0, 100, 200])
UBOUND_ORANGE = np.array([50, 255, 255])

LBOUND = np.array([0, 0, 250])
UBOUND = np.array([60, 100, 255])

LBOUND_WHITE_BGR = np.array([254, 254, 254])
UBOUND_WHITE_BGR = np.array([255, 255, 255])


# END PARAMS

print("OpenCV version: " + cv2.__version__)

def kernel(bgr):
    bgr = cv2.blur(bgr, (11, 11))
    mask_white = cv2.inRange(bgr, LBOUND_WHITE_BGR,UBOUND_WHITE_BGR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  
    # blurred = cv2.blur(hsv, (11, 11))
    mask_bright = cv2.inRange(hsv, LBOUND, UBOUND)
    mask_orange = cv2.inRange(hsv, LBOUND_ORANGE, UBOUND_ORANGE)
    mask = cv2.bitwise_or(mask_orange, mask_bright)
    mask = cv2.bitwise_and(cv2.bitwise_not(mask_white),mask)
    cv2.erode(mask,None,iterations=3)
    cv2.dilate(mask,None,iterations=3)
    cv2.imshow("masked", mask)
    
    points = detector.detect(mask)
    # print(hsv[hsv.shape[0]//2, hsv.shape[1]//2], hsv[hsv.shape[1]//2, hsv.shape[0]//2])
    
    # x = hsv[hsv.shape[0]//2 - 10:hsv.shape[0]//2 + 10, hsv.shape[1]//2-10:hsv.shape[1]//2+10]
    # print([int(np.average(x[:,:,0])),int(np.average(x[:,:,1])),int(np.average(x[:,:,2]))])

    # cv2.circle(bgr, (hsv.shape[1]//2, hsv.shape[0]//2), 40, (0,0,0), 10)
    # cv2.circle(bgr, (hsv.shape[0]//2, hsv.shape[1]//2), 10, (0,0,0), 5)
    # cv2.imshow("bgr", bgr)
    return points

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440 * frameScalingFactor)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960 * frameScalingFactor)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    ### MAIN LOOP

    r, bgr = cap.read()
    
    if r:
        start = time.time()

        points = kernel(bgr)

        end = time.time()

        ### END MAIN LOOP

#        print("FPS: " + str(1.0 / (end - start)))
    
        x = -1.5
        y = -1.5
        # cv2.imshow("yay", bgr)

        best_idx = 0
        biggest_r = 0

        if len(points) > 0:
            # display = cv2.drawKeypoints(bgr, points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("yay", display)
            # x = 0.0
            # y = 0.0
            # for p in points:
            #     x += 0.5 * (p.pt[0] + p.pt[0])
            #     y += 0.5 * (p.pt[1] + p.pt[1])
            # x /= len(points)
            # y /= len(points)

            for p in range(len(points)):
                current_r = points[p].size
                if current_r > biggest_r:
                    biggest_r = current_r
                    best_idx = p

            display = cv2.drawKeypoints(bgr, [points[best_idx]], np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("yay?????????", display)
            x = (points[best_idx].pt[0] - bgr.shape[1] * 0.5) / (bgr.shape[1] * 0.5)
            y = points[best_idx].pt[1] / bgr.shape[0]

        print("x: " + str(x) + "  y: " + str(y))
        sd.putNumber("ball_x|PI_2", x)
        sd.putNumber("ball_y|PI_2", y)

   
    c = cv2.waitKey(1) & 0xFF
    if c == ord('r'):
        LBOUND[0] += 1
    elif c == ord('f'):
        LBOUND[0] -= 1
    elif c == ord('t'):
        LBOUND[1] += 1
    elif c == ord('g'):
        LBOUND[1] -= 1
    elif c == ord('y'):
        LBOUND[2] += 1
    elif c == ord('h'):
        LBOUND[2] -= 1
    if c == ord('u'):
        UBOUND[0] += 1
    elif c == ord('j'):
        UBOUND[0] -= 1
    elif c == ord('i'):
        UBOUND[1] += 1
    elif c == ord('k'):
        UBOUND[1] -= 1
    elif c == ord('o'):
        UBOUND[2] += 1
    elif c == ord('l'):
        UBOUND[2] -= 1
    elif c == ord('q'):
        break
    #print(str(LBOUND) +' '+ str(UBOUND))

cap.release()
cv2.destroyAllWindows()

