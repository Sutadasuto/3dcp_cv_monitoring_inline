import cv2
import re
import time
import os

from threading import Thread
from queue import LifoQueue, Empty
import queue


class DeviceVideoStream:
    # Idea from: https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    def __init__(self, device, stack_size=0):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(device)
        # Version below inspired by https://stackoverflow.com/questions/59309417/cpu-is-100-for-multiprocessing-queue-to-get-frames-from-multiple-cameras
        # self.stream = cv2.VideoCapture(device,cv2.CAP_DSHOW)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.stack = LifoQueue(maxsize=stack_size)
        self.grabbed = False

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        cv2.setNumThreads(1)
        # keep looping infinitely
        while True:

            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if self.stack.full():
                with self.stack.mutex:
                    self.stack.queue.clear()
            else:
                # read the next frame from the file
                (self.grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not self.grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.stack.put(frame)
                time.sleep(1 / (1.5 * self.stream.get(cv2.CAP_PROP_FPS)))

    def read(self):
        # return next frame in the queue
        last_frame = self.stack.get()
        stack_size = self.stack.qsize() + 1
        with self.stack.mutex:
            self.stack.queue.clear()
        return self.grabbed, last_frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

#
# class VideoCapture:
#
#     def __init__(self, name):
#         self.stream = cv2.VideoCapture(name)
#         self.q = queue.Queue()
#         self.stopped = False
#         t = Thread(target=self._reader)
#         t.daemon = True
#         t.start()
#
#     # read frames as soon as they are available, keeping only most recent one
#     def _reader(self):
#         while True:
#             if self.stopped:
#                 return
#             ret, frame = self.stream.read()
#             if not ret:
#                 break
#             if not self.q.empty():
#                 try:
#                     self.q.get_nowait()  # discard previous (unprocessed) frame
#                 except queue.Empty:
#                     pass
#             self.q.put(frame)
#
#     def read(self):
#         return self.q.get()
#
#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True
#
#
# class VideoGet:
#     """
#     Class that continuously gets frames from a VideoCapture object
#     with a dedicated thread.
#     """
#
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False
#
#     def start(self):
#         Thread(target=self.get, args=()).start()
#         return self
#
#     def get(self):
#         while not self.stopped:
#             if not self.grabbed:
#                 self.stop()
#             else:
#                 (self.grabbed, self.frame) = self.stream.read()
#
#     def stop(self):
#         self.stopped = True


class TCPReceiver:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, connection, code_dic):
        self.connection = connection
        self.code_dic = code_dic
        self.current_status = None
        self.stopped = False

    def start(self):
        Thread(target=self.get_messages, args=()).start()
        return self

    def get_messages(self):
        while not self.stopped:
            try:
                input_message = int(self.connection.recv(16))
                print("Last input message: " + self.code_dic[input_message])
            except ValueError:
                break
            self.current_status = input_message

    def stop(self):
        self.stopped = True


def get_parameters(config_file):
    parameters = {}

    with open(config_file, 'r') as parameters_file:
        lines = parameters_file.read().strip().split('\n')

    for line in lines:
        key, value = line.strip().split(", ")
        if key == "video_path":
            try:
                value = int(value)
            except ValueError:
                value = value
        if key == "localhost":
            value = int(value)

        parameters[key] = value

    return parameters


def set_camera(video_input, buffer_size=1):
    vdo = cv2.VideoCapture()
    using_camera = False
    stream = None
    
    try:
        device_id = int(video_input)
        device = True
    except ValueError:
        device = False
        
    if not device:    

        if os.path.isfile(video_input):
            try:
                vdo.open(video_input)
            except IOError:
                raise IOError("%s is not a valid video file." % video_input)
            source_fps = vdo.get(cv2.CAP_PROP_FPS)
        elif os.path.isdir(video_input):
            source_fps = 29.7
            format_str = sorted([f for f in os.listdir(video_input)
                                 if os.path.isfile(os.path.join(video_input, f))
                                 and not f.startswith('.') and not f.endswith('~')], key=lambda f: f.lower())[0]
            numeration = re.findall('[0-9]+', format_str)
            len_num = len(numeration[-1])
            format_str = format_str.replace(numeration[-1], "%0{}d".format(len_num))
            vdo.open(os.path.join(video_input, format_str))
    else:
        try:
            vdo = DeviceVideoStream(device_id, buffer_size).start()
            using_camera = True
            source_fps = vdo.stream.get(cv2.CAP_PROP_FPS)
        except ValueError:
            raise ValueError(
                "{} is neither a valid video file, a folder with valid images nor a proper device id.".format(
                    video_input))

    # return vdo, source_fps, using_camera, stream
    return vdo, source_fps, using_camera
