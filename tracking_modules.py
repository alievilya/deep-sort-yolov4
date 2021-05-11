import itertools
import os
from collections import OrderedDict
from time import gmtime
from time import strftime

import cv2
import imutils


class CountTruth:
    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside


def get_truth(video_name):
    with open('data_files/labels_counted.csv', 'r') as file:
        lines = file.readlines()

    TruthArr = CountTruth(0, 0)
    for line in lines:
        line = line.split(",")
        if line[1] == video_name:
            TruthArr.inside = int(line[2])
            TruthArr.outside = int(line[3])
    return TruthArr


class Counter:
    def __init__(self, counter_in, counter_out, track_id):
        self.fps = 20
        self.max_frame_age_counter = self.fps * 5  # TODO check
        self.max_age_counter = self.fps * 1

        self.people_init = OrderedDict()
        self.people_bbox = OrderedDict()
        self.cur_bbox = OrderedDict()
        self.rat_init = OrderedDict()

        self.age_counter = OrderedDict()
        self.frame_age_counter = OrderedDict()
        self.lost_ids = set()

        # self.dissappeared_frames = OrderedDict()
        self.counter_in = counter_in
        self.counter_out = counter_out
        self.track_id = track_id

    def obj_initialized(self, track_id):
        self.people_init[track_id] = 0

    def delete_person_data(self, track_id):
        del self.people_init[track_id]
        del self.people_bbox[track_id]
        del self.frame_age_counter[track_id]


    def cur_bbox_initialized(self):
        self.cur_bbox = OrderedDict()

    def zero_age(self, track_id):
        self.age_counter[track_id] = 0

    def age_increment(self):
        x = None
        for tr in self.age_counter.keys():
            self.age_counter[tr] += 1
            if self.age_counter[tr] >= self.max_age_counter:
                self.lost_ids.add(tr)
                x = tr
        if self.age_counter.get(x):
            del self.age_counter[x]

    def clear_lost_ids(self):
        self.lost_ids = set()

    def update_identities(self, identities):
        for tr_i in identities:
            if tr_i in self.age_counter.keys():
                if self.frame_age_counter.get(tr_i) is None:
                    self.frame_age_counter[tr_i] = 0
                if self.age_counter.get(tr_i) is None:
                    self.age_counter[tr_i] = 0
                else:
                    self.age_counter[tr_i] = 0
                    self.frame_age_counter[tr_i] += 1
            else:
                # TODO общий счетчик кадров с человеком
                self.age_counter[tr_i] = 0

    def return_lost_ids(self):
        self.age_increment()
        return self.lost_ids

    def get_age(self, track_id):
        return self.age_counter.get(track_id)

    def get_in(self):
        self.counter_in += 1

    def get_out(self):
        self.counter_out += 1

    def show_counter(self):
        return self.counter_in, self.counter_out

    def return_total_count(self):
        return self.counter_in + self.counter_out

    def set_fps(self, frames_per_second):
        self.fps = frames_per_second


class MotionDetector():
    def __init__(self):
        # Number of frames to pass before changing the frame to compare the current
        # frame against
        self.FRAMES_TO_PERSIST = 5
        # Minimum boxed area for a detected motion to count as actual motion
        # Use to filter out noise or small objects
        self.MIN_SIZE_FOR_MOVEMENT = 1400
        # Minimum length of time where no motion is detected it should take
        # (in program cycles) for the program to declare that there is no movement
        self.MOVEMENT_DETECTED_PERSISTENCE = 30
        # Init frame variables
        self.first_frame = None
        self.next_frame = None
        # Init display font and timeout counters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.delay_counter = 0
        self.movement_persistent_counter = 0

    def init_first_frame(self, grayframe):
        if self.first_frame is None:
            self.first_frame = grayframe
    # Read frame
    def find_motion(self, frame):
        self.transient_movement_flag = False
        # Resize and save a greyscale version of the image
        frame = imutils.resize(frame, width=750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # If the first frame is nothing, initialise it
        self.init_first_frame(gray)
        self.delay_counter += 1
        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if self.delay_counter > self.FRAMES_TO_PERSIST:
            delay_counter = 0
            self.first_frame = self.next_frame

        # Set the next frame to compare (the current frame)
        self.next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(self.first_frame, self.next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)

            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > self.MIN_SIZE_FOR_MOVEMENT:
                self.transient_movement_flag = True

                # Draw a rectangle around big enough movements
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if self.transient_movement_flag == True:
            self.movement_persistent_flag = True
            self.movement_persistent_counter = self.MOVEMENT_DETECTED_PERSISTENCE
            return True

        if self.movement_persistent_counter > 0:
            self.movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"
            return False


class Writer():
    def __init__(self):
        self.fps = 3
        self.max_counter_frames_indoor = self.fps * 12
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.counter_frames_indoor = 0
        self.flag_stop_writing = False
        self.flag_writing_video = False
        self.id_inside_door_detected = set()
        self.action_occured = ""
        self.video_name = ""
        self.output_name = ""
        self.noone_in_door = True

    def set_video(self):
        self.max_counter_frames_indoor = self.fps * 15
        hour_greenvich = strftime("%H", gmtime())
        hour_moscow = str(int(hour_greenvich) + 3)
        self.video_name = hour_moscow + strftime(" %M %S", gmtime()) + '.mp4'
        self.output_name = "output/" + self.video_name
        self.output_video = cv2.VideoWriter(self.output_name, self.fourcc, self.fps, (1280, 720))

    def set_id(self, id):
        self.id_inside_door_detected.add(id)

    def start_video(self, id_tracked):
        self.flag_writing_video = True
        self.counter_frames_indoor = 1
        self.set_video()
        self.set_id(id_tracked)

    def continue_opened_video(self, id, seconds=1):
        self.set_id(id)
        self.max_counter_frames_indoor += self.fps * seconds

    def stop_recording(self, action_occured):
        self.flag_stop_writing = True  # флаг об окончании записи
        self.counter_frames_indoor = 0
        self.action_occured = action_occured

    def set_fps(self, frames_per_second):
        self.fps = frames_per_second

    def continue_writing(self, im, flag_anyone_in_door):
        if self.counter_frames_indoor != 0:
            self.counter_frames_indoor += 1
            self.output_video.write(im)
            # return True

        if self.counter_frames_indoor == self.max_counter_frames_indoor:
            if flag_anyone_in_door:
                self.max_counter_frames_indoor += self.fps * 3
            else:
                self.counter_frames_indoor = 0
                self.flag_writing_video = False
                self.flag_stop_writing = False

                if self.output_video.isOpened():
                    self.output_video.release()

                    if os.path.exists(self.output_name):
                        os.remove(self.output_name)

    def stop_writing(self, im):
        if self.flag_stop_writing and self.flag_writing_video:
            self.output_video.write(im)
            if self.video_name[-3:] == "mp4" and self.video_name and os.path.exists(self.output_name):
                self.output_video.release()
                self.flag_stop_writing = False
                return True

        if self.flag_stop_writing and not self.flag_writing_video:
            self.flag_stop_writing = False
            return False

            # send_new_posts(video_name, action_occured)


rect_endpoint_tmp = []
rect_bbox = []
drawing = False


def select_object(img):
    """
    Interactive select rectangle ROIs and store list of bboxes.

    Parameters
    ----------
    img :
           image 3-dim.

    Returns
    -------
    bbox_list_rois : list of list of int
           List of bboxes of rectangle rois.
    """

    # mouse callback function
    bbox_list_rois = []

    def draw_rect_roi(event, x, y, flags, param):

        # grab references to the global variables
        global rect_bbox, rect_endpoint_tmp, drawing

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that drawing is being
        # performed. set rect_endpoint_tmp empty list.
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_endpoint_tmp = []
            rect_bbox = [(x, y)]
            drawing = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # drawing operation is finished
            rect_bbox.append((x, y))
            drawing = False

            # draw a rectangle around the region of interest
            p_1, p_2 = rect_bbox
            cv2.rectangle(img, p_1, p_2, color=(0, 255, 0), thickness=1)
            cv2.imshow('image', img)

            # for bbox find upper left and bottom right points
            p_1x, p_1y = p_1
            p_2x, p_2y = p_2

            lx = min(p_1x, p_2x)
            ty = min(p_1y, p_2y)
            rx = max(p_1x, p_2x)
            by = max(p_1y, p_2y)

            # add bbox to list if both points are different
            if (lx, ty) != (rx, by):
                bbox = [lx, ty, rx, by]
                bbox_list_rois.append(bbox)

        # if mouse is drawing set tmp rectangle endpoint to (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_endpoint_tmp = [(x, y)]

    # clone image img and setup the mouse callback function
    img_copy = img.copy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1200, 600)
    cv2.setMouseCallback('image', draw_rect_roi)

    # keep looping until the 'c' key is pressed
    while True:
        # display the image and wait for a keypress
        if not drawing:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1280, 720)
            cv2.imshow('image', img)
        elif drawing and rect_endpoint_tmp:
            rect_cpy = img.copy()
            start_point = rect_bbox[0]
            end_point_tmp = rect_endpoint_tmp[0]
            cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
            cv2.imshow('image', rect_cpy)

        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord('c'):
            break
    # close all open windows
    cv2.destroyAllWindows()

    return bbox_list_rois


def read_door_info(name='doors_info.csv'):
    door_info = {}
    with open(name, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line_l = line.split(";")
        val = line_l[1][2:-3].split(",")
        for i, v in enumerate(val):
            val[i] = int(v)
        door_info[line_l[0]] = val
    return door_info


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def find_centroid(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2, y2)

    __and__ = intersection

    def difference(self, other):
        inter = self & other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1 < other.x1 < self.x2: xs.add(other.x1)
        if self.x1 < other.x2 < self.x2: xs.add(other.x2)
        if self.y1 < other.y1 < self.y2: ys.add(other.y1)
        if self.y1 < other.y2 < self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
                pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect != inter:
                yield rect

    __sub__ = difference

    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self) == tuple(other)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))


def pairwise(iterable):
    # https://docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def rect_square(x1, y1, x2, y2):
    return abs(x1 - x2) * abs(y1 - y2)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
