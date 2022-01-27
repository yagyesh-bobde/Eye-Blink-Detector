from cv2 import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cam = cv2.VideoCapture('assets/Video.mp4')
detector= FaceMeshDetector(maxFaces=1)

######--------Variables-------########
shape = (480,360)
idlist = [22,23,24,26,110,157,158,159,160,161,130,243]


while 1 :
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT) :
        cam.set(cv2.CAP_PROP_POS_FRAMES,0)

    suc,img= cam.read()
    img,faces = detector.findFaceMesh(img , draw=False)
    face = faces[0]
    for id in idlist :
        cv2.circle(img,face[id],2,(50,255,10),2)

    img = cv2.resize(img, shape)
    cv2.imshow("BLINK COUNTER", img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
         break

cam.release()
cv2.destroyAllWindows()
