import torch
import numpy as np
import cv2
import easyocr
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def detection(frame, model):
    results = model(frame)
    df = results.pandas().xyxy[0]
    
    try:
        maxx = np.argmax(df['confidence'])
        classes = df['name'][maxx]
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        print('detection...............................................')
        return frame, labels, cordinates, classes
    except:
        classes = None
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        print('detection...............................................')
        return frame, labels, cordinates, classes

def plot_boxes(frame ,labels, cordinates, classes):
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cordinates[i]
        if row[4] >= 0.55: 
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
         
            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (3, 201, 136), 2) # BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (3, 201, 136), -1) # for text label background
            cv2.putText(frame, f"{plate_num}, {classes}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2) # text

    return frame

def recognize_plate_easyocr(img, coords,reader,region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] # cropping the number plate
    ocr_result = reader.readtext(nplate) # reading the image using easyocr
    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
    print('recognize_plate_easyocr.........................................')
    return text

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = [] 
    print(ocr_result)
    print('filter_tttttext................................................')
    try:
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
    except:
        pass
    print(plate,'...................................')
    return plate

def main(img_path=None, vid_path=None, vid_out=None, live_path=None):
    print('filter_text................................................')
    model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best.pt',force_reload=True).to(device)
    print('model_load...........................')
    classes = model.names
    print('detecting classes..............................')
    if img_path != None:
        img_out_name = f"./output/result{img_path.split('/')[-1]}"
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame, labels, cordinates, classes = detection(frame, model = model)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if classes:
            frame = plot_boxes(frame, labels, cordinates, classes)
        cv2.namedWindow('window', cv2.WINDOW_NORMAL) # creating a window to show the results

        while True:
            cv2.imshow('window', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f'[INFO] Exiting.....')

                cv2.imwrite(f'{img_out_name}', frame) # if want to save the output
                break
    
    elif vid_path != None:
        print('video is running................................')
        # reading the video 
        cap = cv2.VideoCapture(vid_path)

        if vid_out != None:
            # by default video capture return float insted of int 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        
        # assert cap.isOpened()
        frame_no = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                print(f'[INFO] Working with frame {frame_no}')

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame, labels, cordinates, classes = detection(frame, model = model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if classes:
                    frame = plot_boxes(frame, labels, cordinates, classes)

                cv2.imshow('vid_out', frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        print(f'[INFO] Cleaning up.....')

    elif live_path != None:
        print('video is running................................')
        # reading the video 
        cap = cv2.VideoCapture(0)

        if vid_out != None:
            # by default video capture return float insted of int 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        
        # assert cap.isOpened()
        frame_no = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                print(f'[INFO] Working with frame {frame_no}')

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame, labels, cordinates, classes = detection(frame, model = model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if classes:
                    frame = plot_boxes(frame, labels, cordinates, classes)

                cv2.imshow('vid_out', frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        print(f'[INFO] Cleaning up.....')

        out.release()
        cap.release()

        cv2.destroyAllWindows()
main(img_path='./output/result2.result2.jpg')
# main(vid_path='./test/video.mp4', vid_out='./output/video_out.mp4')
# main(live_path='./output/live.mp4')
