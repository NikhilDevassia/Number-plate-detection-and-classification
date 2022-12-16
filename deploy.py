# importing required libraries
import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
import easyocr

# Defining easyocr
EASYOCR = easyocr.Reader(['en'])
OCR_TH = 0.2 # threshold

def detection(frame, model):
    frame = [frame]
    print(f'[INFO] Detecting.....')
    results = model(frame)

    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, coordinates

def plot_boxes(results, frame, classes):

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f'[INFO] Total {n} detections.....')
    print(f'[INFO] Looping through all detections.....')

    # Looping through all the detections
    for i in range(n):
        cor = cord[i]
        if cor[i] >= 0.55: # threshold value for detection 
            print(f'[INFO] Extracting BBox coordinates...')
            x1, y1, x2, y2 = int(cor[0]*x_shape), int(cor[1]*y_shape), int(cor[2]*x_shape), int(cor[3]*y_shape) # BBox coordinates
            text_d = classes[int(labels[i])]

            coords = [x1, y1, x2, y2]

            plate_num = recognize_plate_easyocr(img=frame, coords=coords, reader=EASYOCR, region_threshold=OCR_TH)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255, 0), -1) # for text label background
            cv2.putText(frame, f'{plate_num}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame
            
def recognize_plate_easyocr(img, coords, reader, region_threshold):
    # seperating coordinates from boxes
    xmin, ymin, xmax, ymax = coords
    
    c_plate = img[int(ymin):int(ymax), int(xmin):int(xmax)] # croping the number plate

    ocr_result = reader.readtext(c_plate)

    text = filter_text(region=c_plate, ocr_result=ocr_result, region_threshold=region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text # return the text in the license plate
    
# filter out wrong detections

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]

    plate = []
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][1]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

# main function
def main(img_path=None, vid_path=None, vid_out=None):
    
    print(f'[info] Loading model.....')
    # loading the custom training model
    model = torch.hub.load('./yolov5', 'custom', source='local', path='best.pt') # loading repo from local 
    Classes = model.names 


    # Detection on image and saving image
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detection(frame, model = model) # detection function

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, Classes)

        # cv2.namedWindow('window', cv2.WINDOW_NORMAL) # creating a window to show the results

        while True:
            cv2.imshow('window', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f'[INFO] Exiting.....')

                cv2.imwrite(f'{img_out_name}', frame) # if want to save the output
                break
    
    # for detection on video
    elif vid_path != None:
        print(f'[INFO] working with video: {vid_path}')

        # reading the video 
        cap = cv2.VideoCapture(vid_path)

        if vid_out:
            # by default video capture return float insted of int 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        # cv2.namedWindow('vid_out', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                print(f'[INFO] Working with frame {frame_no}')

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detection(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = plot_boxes(results, frame, classes=Classes)

                cv2.imshow('vid_out', frame)
                if vid_out:
                    print('[INFO] Saving output video.....')
                    out.write(frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        print(f'[INFO] Cleaning up.....')

        out.release()

        cv2.destroyAllWindows()


main(img_path='./test/1.jpeg')



