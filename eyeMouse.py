import cv2
import numpy as np
import dlib
import math
from scipy.spatial import distance as dist
import pyglet
from pynput.mouse import Button, Controller

mouse = Controller()

cap = cv2.VideoCapture(0)




eye_pos_i=1;
eye_pos=["Mouse Mode","Nothing","Scroll Mode"]
mode_detect=False
text="Nothing"

EYE_AR_THRESH = 0.25  
EYE_AR_CONSEC_FRAMES = 3
MODE_SELECTION_SENSITIVITY=5
   
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
   
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0  

COUNTER_BLINK = 0  
TOTAL_BLINK = 0

EYEBALL_LEFT_COUNTER=0
EYEBALL_RIGHT_COUNTER=0

anchor_pointx=60
anchor_pointy=60




def nothing(x):
    pass



#-------FRAME------------
cv2.namedWindow('face')
cv2.createTrackbar('EYE_AR_THRESH (100^-1)','face',20,50,nothing)
cv2.createTrackbar('MODE_SELECTION_SENSITIVITY','face',20,50,nothing)
cv2.createTrackbar('HELPER','face',1,1,nothing)

#------SOUNDS------------
sound = pyglet.media.load("./sounds/sound.wav", streaming=False)
mouse_mode_sound = pyglet.media.load("./sounds/mouse_mode.wav", streaming=False)
scroll_mode_sound = pyglet.media.load("./sounds/scroll_mode.wav", streaming=False)




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX



def get_blinking_ratio(eye_points, facial_landmarks):
    p1 = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    p2 = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
    p3 = (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y)
    p4 = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    p5 = (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y)
    p6 = (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
 
    line_14 = cv2.line(frame, p1, p4, (0, 255, 0), 2)
    line_26 = cv2.line(frame, p2, p6, (0, 255, 0), 2)
    line_35 = cv2.line(frame, p3, p5, (0, 255, 0), 2)

    line_14_length = dist.euclidean(p1,p4)
    line_26_length = dist.euclidean(p2,p6)
    line_35_length = dist.euclidean(p3,p5)

    eye_aspect_ratio = (line_26_length+line_35_length)/(2*line_14_length)
    return eye_aspect_ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

while True:
    _, frame = cap.read()
    
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    EYE_AR_THRESH=cv2.getTrackbarPos('EYE_AR_THRESH (100^-1)','face')/100
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # CALCULATE RATIO
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        
        #Tracking
        if not mode_detect:
            
            #--------------------------------------------------------------------------------
                                #DETECT BLINK,LEFT WINK,RIGHT EINK
            #--------------------------------------------------------------------------------
            if left_eye_ratio<EYE_AR_THRESH and right_eye_ratio<EYE_AR_THRESH:
                COUNTER_BLINK += 1
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINK += 1
                    #sound.play()
                    mode_detect=not mode_detect
 
                # reset the eye frame counter
                COUNTER_BLINK = 0
        
        
            if left_eye_ratio < EYE_AR_THRESH and right_eye_ratio>EYE_AR_THRESH:  
                COUNTER_LEFT += 1  
            else:  
                 if COUNTER_LEFT >=2:  
                    TOTAL_LEFT += 1  
                    print("Left eye winked")  
                    COUNTER_LEFT = 0
                    mouse.click(Button.right,1)

                
   
            if right_eye_ratio < EYE_AR_THRESH and left_eye_ratio>EYE_AR_THRESH:  
                 COUNTER_RIGHT += 1  
            else:  
                 if COUNTER_RIGHT >= 2:  
                    TOTAL_RIGHT += 1  
                    print("Right eye winked")  
                    COUNTER_RIGHT = 0
                    mouse.click(Button.left,1)
                    
                    
                    
            #--------------------------------------------------------------------------------
                                            #MODES 
            #--------------------------------------------------------------------------------
            
            if eye_pos_i==1:
                #NOTHING
                pass
            elif eye_pos_i==0:
            #--------------------------------------------------------------------------------
                                            #MOUSE MODE
            #--------------------------------------------------------------------------------
                x1=int((landmarks.part(36).x+landmarks.part(39).x)/2)
                y1=int((landmarks.part(37).y+landmarks.part(41).y)/2)
                leftup=(int(anchor_pointx)-20,int(anchor_pointy)-10)
                rightbottom=(int(anchor_pointx)+20,int(anchor_pointy)+10)
                cv2.rectangle(frame,leftup,rightbottom,(255,0,0))
               
                
                vector_length=dist.euclidean((x1,y1),(anchor_pointx,anchor_pointy))
                
                x_mag=y_mag=math.floor(vector_length/6)
                
                print('Speed',x_mag)
                speedx=0;
                speedy=0;
        
                if(x1<leftup[0]):
                    speedx=-x_mag
                    #cv2.putText(frame, "Left", (x1-14, y1), font, 0.7, (0, 0, 255), 2)
                elif(x1>leftup[0] and x1 <rightbottom[0]):
                    speedx=0;
                else:
                    speedx=x_mag;
                    #cv2.putText(frame, "Right", (x1-14, y1), font, 0.7, (0, 0, 255), 3)
        
        
                if(y1<leftup[1]):
                    speedy=-y_mag
                    #cv2.putText(frame, "Up", (x1+14, y1), font, 0.7, (0, 0, 255), 3)
                elif(y1>leftup[1] and y1 <rightbottom[1]):
                    speedy=0;
                else:
                    speedy=y_mag;
                    #cv2.putText(frame, "Down", (x1+14, y1), font, 0.7, (0, 0, 255), 3)
        
                mouse.move(speedx, speedy)
        
                cv2.putText(frame, "Mouse : {}".format(mouse.position), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
                cv2.arrowedLine(frame,(anchor_pointx,anchor_pointy),(x1,y1),1)
                cv2.circle(frame, (x1,y1), 10, (0, 255, 0), thickness=1, lineType=8, shift=0)
            else:
            #--------------------------------------------------------------------------------
                                            #SCROLL MODE 
            #--------------------------------------------------------------------------------
                x1=int((landmarks.part(36).x+landmarks.part(39).x)/2)
                y1=int((landmarks.part(37).y+landmarks.part(41).y)/2)
                leftup=(int(anchor_pointx)-20,int(anchor_pointy)-10)
                rightbottom=(int(anchor_pointx)+20,int(anchor_pointy)+10)
  
                x_mag=y_mag=6
                
                print('Speed',x_mag)
                speedx=0;
                speedy=0;
        
                if(x1<leftup[0]):
                    speedx=-x_mag
                    #cv2.putText(frame, "Left", (x1-14, y1), font, 0.7, (0, 0, 255), 2)
                elif(x1>leftup[0] and x1 <rightbottom[0]):
                    speedx=0;
                else:
                    speedx=x_mag;
                    #cv2.putText(frame, "Right", (x1-14, y1), font, 0.7, (0, 0, 255), 3)
        
        
                if(y1<leftup[1]):
                    speedy=-y_mag
                    #cv2.putText(frame, "Up", (x1+14, y1), font, 0.7, (0, 0, 255), 3)
                elif(y1>leftup[1] and y1 <rightbottom[1]):
                    speedy=0;
                else:
                    speedy=y_mag;
                    #cv2.putText(frame, "Down", (x1+14, y1), font, 0.7, (0, 0, 255), 3)
        
                mouse.scroll(speedx,speedy)
        
                cv2.putText(frame, "Mouse : {}".format(mouse.position), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
                cv2.arrowedLine(frame,(anchor_pointx,anchor_pointy),(x1,y1),1)
                cv2.circle(frame, (x1,y1), 10, (0, 255, 0), thickness=1, lineType=8, shift=0)
   
                
            cv2.putText(frame, "Wink Left : {}".format(TOTAL_LEFT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
            cv2.putText(frame, "Wink Right: {}".format(TOTAL_RIGHT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
            cv2.putText(frame, "Blink: {}".format(TOTAL_BLINK), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
            cv2.putText(frame, "Left Ratio: {}".format(left_eye_ratio), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
            cv2.putText(frame, "Right Ratio: {}".format(right_eye_ratio), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
            cv2.putText(frame, "Blink Ratio: {}".format(blinking_ratio), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, eye_pos[eye_pos_i], (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
        
        # Gaze detection
        else:
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            anchor_pointx=int((landmarks.part(36).x+landmarks.part(39).x)/2)
            anchor_pointy=int((landmarks.part(37).y+landmarks.part(41).y)/2)
            
            MODE_SELECTION_SENSITIVITY=cv2.getTrackbarPos('MODE_SELECTION_SENSITIVITY','face')
            print(EYEBALL_LEFT_COUNTER,EYEBALL_RIGHT_COUNTER)

            if gaze_ratio > 1:
                cv2.putText(frame, "MOUSE", (50, 400), font, 2, (0, 0, 255), 3)
                EYEBALL_RIGHT_COUNTER+=1
                new_frame[:] = (0, 0, 255)
                if EYEBALL_RIGHT_COUNTER>=MODE_SELECTION_SENSITIVITY:
                    EYEBALL_LEFT_COUNTER=0
                    EYEBALL_RIGHT_COUNTER=0
                    eye_pos_i=0
                    #mouse_mode_sound.play()
                    print("hi")
                    mode_detect=not mode_detect
                    
            else:
                new_frame[:] = (255, 0, 0)
                EYEBALL_LEFT_COUNTER+=1
                cv2.putText(frame, "SCROLL", (50, 400), font, 2, (0, 0, 255), 3)
                if EYEBALL_LEFT_COUNTER>=MODE_SELECTION_SENSITIVITY:
                    EYEBALL_LEFT_COUNTER=0
                    EYEBALL_RIGHT_COUNTER=0
                    eye_pos_i=2
                    #scroll_mode_sound.play()
                    mode_detect=not mode_detect
     

        



    
    cv2.imshow("face", frame)
    helper=cv2.getTrackbarPos('HELPER','face')
    if(helper==1 and mode_detect):
        cv2.imshow("New frame", new_frame)
    else:
        cv2.destroyWindow("New frame")

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()