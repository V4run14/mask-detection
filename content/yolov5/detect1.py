# command to run
# python detect1.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.7
from flask import Response
import argparse
import time
from pathlib import Path

from resize import resize_image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os

class Arguments:
    def __init__(self, source, weights, cctv, img_size, conf_thres, iou_thres, device, project, name, augment, classes, agnostic_nms, update):
        self.source = source
        self.weights = weights
        self.cctv = cctv
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.project = project
        self.name = name
        self.augment = augment
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.update = update

def retStream(frame):
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect(save_img=False):
    osource = 'data/images'
    oweights = 'runs/train/yolov5s_results/weights/best.pt'
    occtv = 0
    oimg_size = 416
    oconf_thres = 0.7
    oiou_thres = 0.45
    odevice = ''
    oproject = 'runs/detect'
    oname = 'exp'
    oaugment = True
    oclasses = None
    oagnostic_nms = False
    oupdate = False
    opt = Arguments(osource, oweights, occtv, oimg_size, oconf_thres, oiou_thres, odevice, oproject, oname, oaugment,  oclasses,
                    oagnostic_nms, oupdate)
    check_requirements()
    source, weights, imgsz, cctv = opt.source, opt.weights, opt.img_size, opt.cctv
    save_txt = True
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadImages(source, img_size=imgsz, stride=stride)

    RTSP_URL = "rtsp://admin:Admin123$@10.11.25.53:554/user=admin_password='Admin123$'_channel='Streaming/Channels/'_stream=0.sdp"
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    #cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture('C:/Users/admin/Downloads/videoplayback (2).mp4')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 20)

    if not cap.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    count = 0
    frame_count = 0
    while True:
        _, iframe = cap.read()
        if count % 6 ==0:
            cv2.imshow('RTSP stream', iframe)
            #success, iframe = cap.read()
            '''ret, buffer = cv2.imencode('.jpg', iframe)
            oframe = buffer.tobytes()
            #return Response(retStream(iframe), mimetype='multipart/x-mixed-replace; boundary=frame')
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + oframe + b'\r\n')  # concat frame one by one and show result'''
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_count += 1
            #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            img = resize_image(iframe, 416, cv2.INTER_AREA)

            im0s = img  # BGR

            # Padded resize
            img = letterbox(im0s, 416, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    s, im0, frame = '%g: ' % i, im0s[i].copy(), count
                else:
                    s, im0, frame = '', im0s, count

                # save_path = str(save_dir / 'images')  # img.jpg
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results

                # Save results (image with detections)
                #if save_img:
                    #save_path = save_path + ".jpg"
                    #cv2.imwrite(save_path, im0)

                ret2, buffer2 = cv2.imencode('.jpg', im0)
                oframe2 = buffer2.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + oframe2 + b'\r\n')  # concat frame one by one and show result
                #cv2.imshow('Object Detection', im0)

            # cv2.imwrite("../../CaptureImages/frame%d.jpg" % count, img)

            if cv2.waitKey(1) == 27:
                break
        count += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov5s_results3/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--cctv', type=int, default=0, help='CCTV stream input') # 0 for image read, 1 for cctv read
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()