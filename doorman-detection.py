import os
import socket
from timeit import time

import cv2
import imutils.video
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from easydict import EasyDict as edict

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from draw_enter import select_object
from rectangles import find_centroid, Rectangle, rect_square
from tools import generate_detections as gdet
from tracking_modules import Counter, Writer
from tracking_modules import find_centroid, Rectangle, rect_square, select_object
from yolo import YOLO


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def detect():
    with open("cfg/detection_tracker_cfg.json") as detection_config:
        detect_config = json.load(detection_config)
    print(detect_config["cfg"])

    sent_videos = set()
    video_name = ""
    fpeses = []
    fps = 0

    global lost_ids
    # initial parameters
    # door_array = [596, 76, 650, 295]  #  18 stream
    door_array = [611, 70, 663, 310]
    # around_door_array = [572, 79, 694, 306]  #
    around_door_array = [507, 24, 724, 374]
    low_border = 225
    #
    rect_door = Rectangle(door_array[0], door_array[1], door_array[2], door_array[3])
    door_c = find_centroid(door_array)
    rect_around_door = Rectangle(around_door_array[0], around_door_array[1], around_door_array[2], around_door_array[3])
    # socket
    HOST = "localhost"
    PORT = 8084
    # camera info
    save_img = True

    imgsz = detect_config["img_size"]  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = detect_config["output"], detect_config["source"], detect_config["weights"], \
                                           detect_config["half"], detect_config["view_img"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    counter = Counter(counter_in=0, counter_out=0, track_id=0)
    VideoHandler = Writer()
    yolo = YOLO()

    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 1.0

    output_format = 'mp4'

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    error_values = []
    check_gpu()
    files = sorted(os.listdir('data_files'))
    for video_name in files[0]:
        file_path = detect_config["source"]
        print("opening video: {}".format(video_name))
        # file_path = os.path.join('data_files/videos', video_name)

        output_name = 'save_data/out_' + video_name[0:-3] + output_format

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            video_capture = cv2.VideoCapture(file_path)
            if writeVideo_flag:
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_name, fourcc, 15, (w, h))
                frame_index = -1
            fps = 0.0
            fps_imutils = imutils.video.FPS().start()

            while True:
                ret, im0 = video_capture.read()
                flag_anyone_in_door = False

                ratio_detection = 0

                classes = None if detect_config["needed_classes"] == "None" else detect_config["needed_classes"]
                # Process detections
                lost_ids = counter.return_lost_ids()

                t1 = time.time()

                image = Image.fromarray(im0[..., ::-1])  # bgr to rgb
                boxes, confidence, classes = yolo.detect_image(image)

                features = encoder(im0, boxes)
                detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                              zip(boxes, confidence, classes, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.cls for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for i, det in enumerate(detections):  # detections for image i
                    bbox = det.to_tlbr()
                    # if show_detections and len(classes) > 0:
                    #     score = "%.2f" % (det.confidence * 100) + "%"
                    #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)

                    if len(door_array) != 4 or len(around_door_array) != 4:
                        door_array = select_object(im0)
                        print(door_array)
                    # counter.update_identities(identities)
                    # Stream results

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()

                    if track.track_id not in counter.people_init or counter.people_init[track.track_id] == 0:
                        counter.obj_initialized(track.track_id)
                        rect_detection = Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                        inter_detection = rect_detection & rect_around_door
                        if inter_detection:
                            inter_square_detection = rect_square(*inter_detection)
                            cur_square_detection = rect_square(*rect_detection)
                            try:
                                ratio_detection = inter_square_detection / cur_square_detection
                            except ZeroDivisionError:
                                ratio_detection = 0
                            #  чел первый раз в контуре двери
                        if ratio_detection > 0.2:
                            if VideoHandler.counter_frames_indoor == 0:
                                #     флаг о начале записи
                                VideoHandler.start_video(track.track_id)
                            flag_anyone_in_door = True

                        elif ratio_detection > 0.2 and track.track_id not in VideoHandler.id_inside_door_detected:
                            VideoHandler.continue_opened_video(id=track.track_id, seconds=3)
                            flag_anyone_in_door = True

                        # elif ratio_detection > 0.6 and counter.people_init.get(id_tracked) == 1:
                        #     VideoHandler.continue_opened_video(id=id_tracked, seconds=0.005)

                        if track.track_id not in counter.people_init or counter.people_init[track.track_id] == 0:
                            counter.obj_initialized(track.track_id)
                            rect_head = Rectangle(bbox[0], bbox[1], bbox[2],
                                                  bbox[3])
                            intersection = rect_head & rect_door
                            if intersection:
                                intersection_square = rect_square(*intersection)
                                head_square = rect_square(*rect_head)
                                rat = intersection_square / head_square
                                if rat >= 0.4:
                                    #     was initialized in door, probably going out of office
                                    counter.people_init[track.track_id] = 2
                                elif rat < 0.4:
                                    #     initialized in the corridor, mb going in
                                    counter.people_init[track.track_id] = 1
                            else:
                                # res is None, means that object is not in door contour
                                counter.people_init[track.track_id] = 1
                            counter.frame_age_counter[track.track_id] = 0

                            counter.people_bbox[track.track_id] = bbox

                        counter.cur_bbox[track.track_id] = bbox
                    # first appearence of object with id=track.id

                id_get_lost = [track.track_id for track in tracker.tracks if track.time_since_update >= 35]

                vals_to_del = []

                for val in counter.people_init.keys():
                    # check bbox also
                    inter = 0
                    cur_square = 0
                    ratio = 0
                    cur_c = find_centroid(counter.cur_bbox[val])
                    centroid_distance = np.sum(np.array([(door_c[i] - cur_c[i]) ** 2 for i in range(len(door_c))]))

                    # init_c = find_centroid(counter.people_bbox[val])
                    # vector_person = (cur_c[0] - init_c[0],
                    #                  cur_c[1] - init_c[1])

                    rect_cur = Rectangle(counter.cur_bbox[val][0], counter.cur_bbox[val][1],
                                         counter.cur_bbox[val][2], counter.cur_bbox[val][3])
                    inter = rect_cur & rect_door

                    if val in id_get_lost and counter.people_init[val] != -1:
                        if inter:
                            inter_square = rect_square(*inter)
                            cur_square = rect_square(*rect_cur)
                            try:
                                ratio = inter_square / cur_square
                            except ZeroDivisionError:
                                ratio = 0
                        # if vector_person < 0 then current coord is less than initialized, it means that man is going
                        # in the exit direction

                        if counter.people_init[val] == 2 \
                                and ratio < 0.4 and centroid_distance > 5000:  # vector_person[1] > 50 and
                            print('ratio out: {}\n centroids: {}\n'.format(ratio, centroid_distance))
                            counter.get_out()
                            counter.people_init[val] = -1
                            VideoHandler.stop_recording(action_occured="вышел из кабинета")

                            vals_to_del.append(val)

                        elif counter.people_init[val] == 1 \
                                and ratio >= 0.4 and centroid_distance < 1000:  # vector_person[1] < -50 and
                            print('ratio in: {}\n centroids: {}\n'.format(ratio, centroid_distance))
                            counter.get_in()
                            counter.people_init[val] = -1
                            VideoHandler.stop_recording(action_occured="зашел внутрь")
                            vals_to_del.append(val)

                        lost_ids.remove(val)

                    # TODO maybe delete this condition
                    elif counter.frame_age_counter.get(val, 0) >= counter.max_frame_age_counter \
                            and counter.people_init[val] == 2:
                        if inter:
                            inter_square = rect_square(*inter)
                            cur_square = rect_square(*rect_cur)
                            try:
                                ratio = inter_square / cur_square
                            except ZeroDivisionError:
                                ratio = 0

                        if ratio < 0.2 and centroid_distance > 10000:  # vector_person[1] > 50 and
                            counter.get_out()
                            print('ratio out max frames: ', ratio)
                            counter.people_init[val] = -1
                            VideoHandler.stop_recording(action_occured="вышел")
                            vals_to_del.append(val)
                        counter.age_counter[val] = 0

                    counter.clear_lost_ids()

                for valtodel in vals_to_del:
                    counter.delete_person_data(track_id=valtodel)

                ins, outs = counter.show_counter()
                cv2.rectangle(im0, (0, 0), (250, 50),
                              (0, 0, 0), -1, 8)

                cv2.rectangle(im0, (int(door_array[0]), int(door_array[1])),
                              (int(door_array[2]), int(door_array[3])),
                              (23, 158, 21), 3)

                cv2.rectangle(im0, (int(around_door_array[0]), int(around_door_array[1])),
                              (int(around_door_array[2]), int(around_door_array[3])),
                              (48, 58, 221), 3)

                cv2.putText(im0, "in: {}, out: {} ".format(ins, outs), (10, 35), 0,
                            1e-3 * im0.shape[0], (255, 255, 255), 3)

                cv2.line(im0, (door_array[0], low_border), (880, low_border), (214, 4, 54), 4)

                if VideoHandler.stop_writing(im0):
                    # send_new_posts(video_name, action_occured)
                    sock.sendall(bytes(VideoHandler.video_name + ":" + VideoHandler.action_occured, "utf-8"))
                    data = sock.recv(100)
                    print('Received', repr(data.decode("utf-8")))
                    sent_videos.add(VideoHandler.video_name)
                    with open('data_files/logs2.txt', 'a', encoding="utf-8-sig") as wr:
                        wr.write(
                            'video {}, man {}, centroid {} '.format(VideoHandler.video_name,
                                                                    VideoHandler.action_occured,
                                                                    centroid_distance))

                    VideoHandler = Writer()
                    VideoHandler.set_fps(fps)

                else:
                    VideoHandler.continue_writing(im0, flag_anyone_in_door)

                if view_img:
                    cv2.imshow('im0', im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                delta_time = (time.time() - t1)
                # t2_ds = time.time()
                # print('%s Torch:. (%.3fs)' % (s, t2 - t1))
                # print('Full pipe. (%.3fs)' % (t2_ds - t0_ds))
                if len(fpeses) < 30:
                    fpeses.append(round(1 / delta_time))
                elif len(fpeses) == 30:
                    # fps = round(np.median(np.array(fpeses)))
                    fps = np.median(np.array(fpeses))
                    # fps = 20
                    print('fps set: ', fps)
                    VideoHandler.set_fps(fps)
                    counter.set_fps(fps)
                    fpeses.append(fps)
                    motion_detection = True
                else:
                    print('\nflag writing video: ', VideoHandler.flag_writing_video)
                    print('flag stop writing: ', VideoHandler.flag_stop_writing)
                    print('flag anyone in door: ', flag_anyone_in_door)
                    print('counter frames indoor: ', VideoHandler.counter_frames_indoor)
                # fps = 20


# python detect.py --cfg cfg/csdarknet53s-panet-spp.cfg --weights cfg/best14x-49.pt --source 0
import json

if __name__ == '__main__':
    detect()
