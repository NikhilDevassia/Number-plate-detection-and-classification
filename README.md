License Plate Detection and Recognition
This script is a demonstration of how to detect and recognize license plates in images or video using PyTorch, NumPy, OpenCV and easyocr.

Dependencies
PyTorch
NumPy
OpenCV
easyocr
Usage
The script takes several inputs:

img_path: path to an image file for detection and recognition
vid_path: path to a video file for detection and recognition
vid_out: path to save the output video with detection and recognition results
live_path: path to a live camera feed for detection and recognition
Detection and Recognition
The script uses a pre-trained object detection model from PyTorch Hub to detect license plates in an image or video. The detection results are then passed to the recognize_plate_easyocr function which uses easyocr's Reader class to recognize the text on the license plate. The recognition results are then overlayed on top of the detection results and returned.

Filtering
The function filter_text is used to filter out any recognition results that are less than a certain threshold of the total rectangle size. This is to ensure that only clear and accurate recognition results are returned.

Note
The script uses a specific version of the YOLOv5 model, so it is important to use the same version when loading the model.

Conclusion
This script is a demonstration of how to use PyTorch, NumPy, OpenCV and easyocr for license plate detection and recognition. The script can be further modified to suit specific use cases and requirements.
