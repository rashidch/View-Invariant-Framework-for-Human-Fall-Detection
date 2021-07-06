import cv2


cap = cv2.VideoCapture('examples/demo/test/Angle_A.mp4')

if cap.isOpened()==False:
    print("Error opening video stream or file")

fwidth  = int(cap.get(3))
fheight = int(cap.get(4))

writer = cv2.VideoWriter('outputs/camtest/A.avi',cv2.VideoWriter_fourcc('M','J','P','G'),15, (fwidth,fheight))
while (cap.isOpened):
    success,frame = cap.read()
    if success==True:
        writer.write(frame.astype('uint8'))
        cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()



