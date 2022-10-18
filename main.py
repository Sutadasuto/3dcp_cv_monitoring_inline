import time

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Disable eager execution to avoid conflicts with the encoder-decoder architecture

import argparse
import numpy as np

from distutils.util import strtobool
from util import *

import data

from models.available_models import get_models_dict
from texture.classify import TextureClassifier

import socket

# A brief natural language dictionary for socket communication between Matlab and Python
tcp_dic = {
    None: "none",
    -1: "finish",
    0: "matlab_waiting",
    1: "writing_ready",
    2: "matlab_processing",
    3: "no_more_frames"
}

# Used to create neural network architecture
models_dict = get_models_dict()


def main(args):
    script_directory = os.path.abspath(os.path.dirname(__file__))
    parameters = get_parameters(os.path.join(script_directory, "config_files", "cam_and_com_config")) # Get video source, network weights, and localhost port

    # The outputs of the script will be saved here
    results_path = os.path.join(script_directory, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    print("Preparing neural network")
    # Get the pre-trained weights of U-VGG19
    with open(os.path.join(script_directory, "config_files", "config"), 'r') as parameters_file:
        lines = parameters_file.read().strip().split('\n')
    for line in lines:
        key, value = line.strip().split(", ")
        if key == "weights_path":
            weights_path = value

    # Deal with GPU usage
    print("Preparing neural network for interlayer segmentation")
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Used for memory error in RTX2070
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Initialize U-VGG19 for interlayer segmentation
    input_size = (None, None)
    model = models_dict["uvgg19"]((input_size[0], input_size[1], 1))
    rgb_preprocessor = data.get_preprocessor(model)
    model.load_weights(weights_path)
    model.compile(loss="bce")

    # Prepare the texture CNN in a parallel thread
    texture_classifier = TextureClassifier()

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
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
        input_message = listener.current_status
        if using_camera:
            grabbed, ori_im = vdo.read()

        if tcp_dic[input_message] == "matlab_waiting":
            if not using_camera:
                grabbed, ori_im = vdo.read()
            listener.current_status = None

            if not grabbed:
                print("No frame grabbed.")
                status = 3
                connection.sendall(status.to_bytes(1, "big"))
                break

            extension = ".tiff"  # Extension used to save the captured image. To be read by Matlab
            # Save image to the results folder
            image_path = os.path.join(results_path, "input_image" + extension)
            cv2.imwrite(image_path, ori_im)

            # Detect the interlayer lines
            start = time.time()
            [im, pred] = data.test_image_from_path(model, image_path, rgb_preprocessor=rgb_preprocessor)
            or_shape = ori_im.shape
            pred = pred[:or_shape[0], :or_shape[1], 0]
            print("Interlayer lines segmented in {:.3f} s".format(time.time() - start))

            # Save the segmentation in the results folder
            segmentation_path = os.path.join(results_path, "interlayer_lines.png")
            cv2.imwrite(segmentation_path, np.where(pred >= 0.5, 255, 0))
            status = 1
            connection.sendall(status.to_bytes(1, "big"))  # Tell Matlab the images have been written
            while True: # In case Matlab doesn't provide texture images
                if tcp_dic[listener.current_status] == "matlab_waiting":  # There was an error processing the previous image and Matlab is asking for a new one
                    break  # Get out to grab new image
                else:
                    if texture_classifier.matlab_flag_exists():  # If Matlab sent texture images
                        texture_classifier.classify_textures()  # Classify the textures sent by Matlab before getting a new frame
                        break
                    else:
                        time.sleep(0.1)  # To reduce number of instructions
        elif tcp_dic[input_message] == "finish":
            print("Finish instruction received from Matlab")
            break

    if using_camera:
        vdo.stop()  # Close camera properly before finishing the program

    # Be sure to properly clear the socket
    listener.stop()
    connection.close()
    sock.close()



def parse_args(args=None):
    parser = argparse.ArgumentParser()
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
