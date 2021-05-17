import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import cv2
import numpy as np
import os
import re
import time

from distutils.util import strtobool
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError
from util import *

import data

from models.available_models import get_models_dict

import socket
import sys

import multiprocessing
from queue import LifoQueue


tcp_dic = {
    -1: "finish",
    0: "matlab_waiting",
    1: "writing_ready",
    2: "matlab_processing",
    3: "no_more_frames"
}

models_dict = get_models_dict()


def main(args):
    max_res = args.max_res
    parameters = get_parameters(args)
    
    print("Preparing neural network")
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Used for memory error in RTX2070
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    input_size = (None, None)

    model = models_dict["uvgg19"]((input_size[0], input_size[1], 1))
    rgb_preprocessor = data.get_preprocessor(model)
    model.load_weights(parameters["weights_path"])

    print("Setting video input")
    # vdo, source_fps, using_camera, stream = set_camera(parameters["video_path"], 30)
    vdo, source_fps, using_camera = set_camera(parameters["video_path"], 30)
    # using_camera = True
    # stream = cv2.VideoCapture(0)
    # stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # queue_from_cam = multiprocessing.Queue(maxsize=1)
    # queue_from_cam = LifoQueue()
    # grabbed = False
    #
    # def cam_loop(queue_from_cam):
    #     global grabbed
    #     print('initializing cam')
    #     cap = cv2.VideoCapture(0)
    #     while True:
    #         if queue_from_cam.full():
    #             with queue_from_cam.mutex:
    #                 queue_from_cam.queue.clear()
    #         grabbed, img = cap.read()
    #         queue_from_cam.put(img)
    #         time.sleep(1/1.5*cap.get(cv2.CAP_PROP_FPS))
    #
    # cam_process = multiprocessing.Process(target=cam_loop, args=(queue_from_cam,))
    # cam_process.start()
    # using_camera = True
    # source_fps = 25
    #
    # print("bliu")
    # ori_im = queue_from_cam.get()

    grabbed, ori_im = vdo.read()
    if not grabbed:
        raise RuntimeError("Unable to read first frame from video input.")

    print("Waiting for a connection")
    server_address = ('localhost', parameters["localhost"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(1)
    try:
        connection, client_address = sock.accept()
        print("Connection ready. Start!")
    except:
        sock.close()
    listener = TCPReceiver(connection, tcp_dic).start()
    while True:

        grabbed, ori_im = vdo.read()

        if not grabbed:
            print("No frame grabbed.")
            status = 3
            connection.sendall(status.to_bytes(1, "big"))
            break

        input_message = listener.current_status
        if input_message is None:
            if not using_camera:
                time.sleep(1/source_fps)
            continue

        status = 0
        if tcp_dic[input_message] == "matlab_waiting":
            listener.current_status = None
            if ori_im.shape[1] > max_res:
                scale_factor = max_res/ori_im.shape[1]
                ori_im = cv2.resize(ori_im,
                                    (
                                        round(scale_factor*ori_im.shape[1]),
                                        round(scale_factor * ori_im.shape[0])
                                    )
                                    )
            ori_im = data.manual_padding(ori_im, n_pooling_layers=4)
            prediction = model.predict(
                rgb_preprocessor(cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))[None, ...], verbose=1)[0, ...]
            cv2.imwrite(os.path.join("tmp", "img.jpg"), ori_im)
            cv2.imwrite(os.path.join("tmp", "img_gt.png"), 255*np.where(prediction > 0.5, 1.0, 0.0))
            status = 1
        elif tcp_dic[input_message] == "finish":
            print("Finish instruction received from Matlab")
            break
        else:
            if not using_camera:
                time.sleep(1/source_fps)

        connection.sendall(status.to_bytes(1, "big"))

    if using_camera:
        vdo.stop()

    listener.stop()
    connection.close()
    sock.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters_file", type=str, default="config")
    parser.add_argument("--max_res", type=int, default=1280)
    parser.add_argument("--gpu", type=str, default=True)

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)