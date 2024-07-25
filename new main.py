import cv2
thres=0.5


cap=cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,1080)



#loading class names
classNames=[]
classFile='coco.names'
with open(classFile,'r') as f:
    classNames=f.read().rstrip('\n').split('\n')

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'

#creating detection model
net=cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

#main loop
while True:
    success,img=cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) !=0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=1)
            cv2.putText(img,classNames[classId-1].upper(),(box[0],box[1]),cv2.FONT_HERSHEY_COMPLEX,1/2,(0,50,0),1)

            cv2.putText(img,str(round(confidence*100,2)),(box[0]+150,box[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,50,0),1)

    cv2.imshow("Output",img)
    cv2.waitKey(10)