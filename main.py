from cv2 import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot


cam = cv2.VideoCapture('assets/Video.mp4')
detector = FaceMeshDetector(maxFaces=1)
plot = LivePlot(480, 360, [20, 50], invert=True)

######--------Variables-------########
shape = (480,360)
idlist = [22,160,161,24,26,110,157,158,
          159,23,130,243] # verticle and horizontal points
color = (255,0,255)
ratiolist = []
sen = 3 # controls the flucutation in plot
counter = 0 # to count the frame no
blinkcounter = 0


while True :
    #__________Restart the video____________#
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT) :
        cam.set(cv2.CAP_PROP_POS_FRAMES,0)

    #________Read the frame________________#
    suc,img= cam.read()

    #________face detection_________________#
    img,faces = detector.findFaceMesh(img , draw=False)
    face = faces[0]

    #________Circle the eye________#
    for id in idlist :
        cv2.circle(img,face[id],2,color,2)

    #______Calculate the eye measurements to calculate when blinked________#
    leftUp = face[159]
    leftDown = face[23]
    left = face[130]
    right = face[243]
    lenghtVer, _ = detector.findDistance(leftUp, leftDown)
    lenghtHor, _ = detector.findDistance(left, right)
    cv2.line(img,leftUp,leftDown,color,2)
    cv2.line(img, left, right, color, 2)


    ratio = (lenghtVer/lenghtHor)*100
    ratiolist.append(ratio)
    if len(ratiolist) == sen :
        ratiolist.pop(0)
    ratioAvg = (sum(ratiolist))/len(ratiolist)

    if ratioAvg < 35 and counter == 0 :
        blinkcounter+= 1
        counter=1
        color = (0,255,0)

    if counter != 0 :
        counter+=1
        if counter ==10 :
            counter = 0
            color = (255,0,255)

    cvzone.putTextRect(img, f'Blink Count: {blinkcounter}', (50, 100),
                       colorR=color)

    img = cv2.resize(img, shape)
    imgPlot = plot.update(ratioAvg, color)

    stackedImage = cvzone.stackImages([img,imgPlot],2,1)
    cv2.imshow("BLINK COUNTER", stackedImage)
    if cv2.waitKey(30) & 0xFF == ord('q') :
         break

cam.release()
cv2.destroyAllWindows()
