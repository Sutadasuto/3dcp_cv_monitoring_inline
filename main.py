import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Disable eager execution to avoid conflicts with the encoder-decoder architecture

import argparse
import cv2
import numpy as np
import os

from distutils.util import strtobool
from util import *

import data

from models.available_models import get_models_dict

import socket

# A brief natural language dictionary for socket communication between Matlab and Python
tcp_dic = {
    -1: "finish",
    0: "matlab_waiting",
    1: "writing_ready",
    2: "matlab_processing",
    3: "no_more_frames"
}

# Used to create neural network architecture
models_dict = get_models_dict()


def main(args):
    max_res = args.max_res
    parameters = get_parameters(args) # Get video source, network weights, and localhost port
    
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
    vdo, source_fps, using_camera = set_camera(parameters["video_path"], 30)
    # Test camera or input source
    grabbed, ori_im = vdo.read()
    if not grabbed:
        raise RuntimeError("Unable to read first frame from video input.")

    print("Waiting for a connection")
    # Create TCP server
    server_address = ('localhost', parameters["localhost"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    # Server ready, waiting for communication from Matlab
    sock.listen(1)
    try:
        connection, client_address = sock.accept()
        print("Connection ready. Start!")
    except:
        sock.close()
    listener = TCPReceiver(connection, tcp_dic).start()  # Put TCP listening in a different thread to don't stop the workflow
    # Begin the infinite loop for reading and processing images
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
                time.sleep(1/source_fps)  # Use a delay to emulate real-time acquisition
            continue

        status = 0
        if tcp_dic[input_message] == "matlab_waiting":
            listener.current_status = None
            # Resize image if needed to fit to max_res
            if ori_im.shape[1] > max_res:
                scale_factor = max_res/ori_im.shape[1]
                ori_im = cv2.resize(ori_im,
                                    (
                                        round(scale_factor*ori_im.shape[1]),
                                        round(scale_factor * ori_im.shape[0])
                                    )
                                    )
            ori_im = data.manual_padding(ori_im, n_pooling_layers=4) # To avoid dimension match issues during prediction
            # Predict input image an save it along with its prediction
            prediction = model.predict(
                rgb_preprocessor(cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))[None, ...], verbose=1)[0, ...]
            cv2.imwrite(os.path.join("tmp", "img.png"), ori_im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(os.path.join("tmp", "img_gt.png"), 255*np.where(prediction > 0.5, 1.0, 0.0))
            status = 1
        elif tcp_dic[input_message] == "finish":
            print("Finish instruction received from Matlab")
            break
        else:
            if not using_camera:
                time.sleep(1/source_fps)  # Use a delay to emulate real-time acquisition

        connection.sendall(status.to_bytes(1, "big"))  # Tell Matlab the images have been written

    if using_camera:
        vdo.stop()  # Close camera properly before finishing the program

    # Be sure to properly clear the socket
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


# Run the script
if __name__ == "__main__":
    args = parse_args()  # Parse user arguments
    main(args)  # Run main function with user arguments
