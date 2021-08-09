import cv2
import os
import numpy
import argparse
import time
import glob



def getFrames():
    path = 'C:\\Users\\rashi\\Documents\\FallDataset\\RawVideos'
    files = glob.glob(path+'/*/*/*avi')
    nameFormat = "{:05d}.png"
    currentFrame = 0
    for file in files:
        placeName = file.split('\\')[-3]
        videoId = file.split('\\')[-1]
        print(videoId)
        videoId = videoId.split('.')[0]
        fpath = os.path.join(path,placeName,'Videos',videoId)
        if not os.path.exists(fpath):
            os.makedirs(fpath)   
        print('Processing File:',file)    
        cap = cv2.VideoCapture(file)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                Outpath = os.path.join(fpath,str(nameFormat.format(currentFrame)))
                cv2.imwrite(Outpath,frame)
                print(nameFormat.format(currentFrame))
                currentFrame = currentFrame+1
            else:
                print('Handle not captured')
                break
        time.sleep(1.0)
        cap.release()
        currentFrame=0

def AnnotateFrames():
    
    path = 'C:\\Users\\rashi\\Documents\\FallDataset\\RawVideos'
    tfiles = glob.glob(path+'/*/*/*txt')
    #vfiles = glob.glob(path+'/*/*/*avi')
    nameFormat = "{:05d}.png"
    localFrame  = 0
    globalFrame = 0 
    for file in tfiles:
        placeName = file.split('\\')[-3]
        videoId = file.split('\\')[-1]
        print(videoId)
        videoId = videoId.split('.')[0]
        with open(file) as file:
            start,end = next(file).strip(),next(file).strip()
            if len(start)<4 or len(end)<4:
                start,end = int(start),int(end)
            else: 
                #print(start,end)
                start,end = 0,0
        bound = [ids for ids in range (start,end,1)]
        print(bound)
        time.sleep(1.0)
        vfile = os.path.join(path,placeName,'Videos',videoId+'.avi') 
        print('Processing File:',vfile)    
        if len(bound)==0:
            print('Zero Frame')
            time.sleep(1.0)
        cap = cv2.VideoCapture(vfile)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if localFrame  in bound:
                    fpath = os.path.join(path,placeName,'Videos','Fall') 
                    if not os.path.exists(fpath):
                        os.makedirs(fpath) 
                    Outpath = os.path.join(fpath,str(nameFormat.format(globalFrame)))
                    cv2.imwrite(Outpath,frame)
                    print(nameFormat.format(localFrame))
                elif localFrame not in bound:
                    fpath = os.path.join(path,placeName,'Videos','NotFall') 
                    if not os.path.exists(fpath):
                        os.makedirs(fpath) 
                    Outpath = os.path.join(fpath,str(nameFormat.format(globalFrame)))
                    cv2.imwrite(Outpath,frame)
                    #print(nameFormat.format(localFrame))
                localFrame = localFrame+1
                globalFrame = globalFrame+1
            else:
                print('Handle not captured')
                break
        time.sleep(1.0)
        cap.release()
        localFrame=0
        
def getAnnoframes():
    path = '/home/rasho/Falling-Person-Detection-based-On-AlphaPose/input/le2i_annotated/train'
    files = glob.glob(path+'/*/*mp4')
    nameFormat = "{:05d}.png"
    currentFrame = 0
    for file in files:
        label = file.split('/')[-2]
        print(label)
        fpath = os.path.join(path,label+'_frames')
        if not os.path.exists(fpath):
            os.makedirs(fpath)   
        print('Processing File:',file)    
        cap = cv2.VideoCapture(file)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                Outpath = os.path.join(fpath,str(nameFormat.format(currentFrame)))
                cv2.imwrite(Outpath,frame)
                print(nameFormat.format(currentFrame))
                currentFrame = currentFrame+1
            else:
                print('Handle not captured')
                break
        time.sleep(1.0)
        cap.release()
    

if __name__=='__main__':
    getAnnoframes()