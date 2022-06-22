import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime=0

# to draw on our faces
mpDraw = mp.solutions.drawing_utils

# We need to load facemesh and create an object for that
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3) #facemesh accepts only RGB imgs

drawSpec = mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)

while True:
    success, img =cap.read()

    # We must see the result but first if a fundamental step: convert the color format. Opencv uses BGR instead of RBG.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # to display the results
    if results.multi_face_landmarks:
        # loop through many faces if detected
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                # image height, width and channels
                ih, iw, ic = img.shape
                #normalise and convert to pixels
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)

    # write the frame
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
