
'''
import cv2

if __name__ == "__main__":
    # Opens the Video file
    cap= cv2.VideoCapture('examples/demo/1.mp4')
    i=0
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('examples/demo/vis/'+str(i)+'.png',frame)
        print('...')
        i+=1
    cap.release()
    cv2.destroyAllWindows()


brightness_4

'''

# Program To Read video 
# and Extract Frames 
import cv2 
  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        if success:
            # Saves the frames with frame-count 
            cv2.imwrite("examples/demo/vis"+str(count)+".jpg", image) 
  
            count += 1
        else:
            break
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("examples/demo/1.mp4") 