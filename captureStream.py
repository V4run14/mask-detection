import cv2
import numpy as np
import os

def resize_image(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)

# RTSP_URL = 'rtsp://admin:Admin123$@10.11.25.53:554/Streaming/Channels/'

RTSP_URL = "rtsp://admin:Admin123$@10.11.25.60:554/user=admin_password='Admin123$'_channel='Streaming/Channels/'_stream=0.sdp"

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#cap = cv2.VideoCapture('C:/Users/admin/Downloads/videoplayback.mp4')

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

count = 0
while True:
    _, frame = cap.read()
    cv2.imshow('RTSP stream', frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    if count % 100 == 0:
        #resized = resize_image(frame, 416, cv2.INTER_AREA)
        cv2.imwrite("CaptureImages/frame%d.jpg" % count, frame)

    if cv2.waitKey(30) == 27:
        break
    count += 1
cap.release()
cv2.destroyAllWindows()