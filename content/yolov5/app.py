from flask import Flask, render_template, Response
import cv2
from detect1 import *

app = Flask(__name__)

RTSP_URL = "rtsp://admin:Admin123$@10.11.25.60:554/user=admin_password='Admin123$'_channel='Streaming/Channels/'_stream=0.sdp"
#cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

#camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)




@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #detect()



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)