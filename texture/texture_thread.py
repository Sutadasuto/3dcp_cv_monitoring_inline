from texture import classify
from threading import Thread


import os
import warnings
from time import sleep, time

from texture.data import calculate_features
from joblib import load
from texture.utils import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import user parameters from config file
from texture import config

class TextureClassifier:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        self.status = "waiting"
        self.stopped = False
        self.config_dict = config.__dict__  # Parameters for feature extraction
        self.photo_dir = "images"

    def start(self):
        Thread(target=self.classify_textures, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def classify_textures(self):
        self.classify_textures()

    def classify_textures(self):

        # Load preliminar variables
        script_dir = os.path.split(os.path.realpath(__file__))[0]
        scaler = load(os.path.join(script_dir, "data_scaler.joblib"))
        enc = load(os.path.join(script_dir, "class_encoder.joblib"))
        class_names = enc.categories_[0]
        model_ready = False

        try:
            best_feats = load(os.path.join(script_dir, "best_features.joblib"))
        except FileNotFoundError:
            best_feats = None
            warnings.warn("There is no file with to read the list of best features. Using all features for prediction.")

        while True:
            print("Waiting to classify textures...")
            while not os.path.exists(os.path.join(script_dir, self.photo_dir, "matlab_flag")) or os.path.exists(
                    os.path.join(script_dir, self.photo_dir, "python_flag")):
                sleep(0.1)
                if self.stopped:
                    return 0
            print("Textures have been received.")

            start = time()
            X, names, feature_names = calculate_features(os.path.join(script_dir, self.photo_dir), **self.config_dict)
            print("Features extracted in {:.3f} s".format(time() - start))

            # Data pre-processing as done during training
            X = scaler.transform(X)

            # Select the best features used for training (if available)
            if best_feats is not None:
                X = X[:, best_feats]

            ### Load classifier (if it is not yet ready)
            if not model_ready:
                clf = create_model(input_size=X.shape[-1], output_size=len(class_names), conv_size=6, dropout=0.3)
                clf.load_weights(os.path.join(script_dir, "weights.hdf5"))

            start = time()
            preds = clf.predict(X[:, None, :, None])
            all_outputs = ["file," + ",".join(class_names.tolist())]

            for i, pred in enumerate(preds):
                output = []
                for option in range(pred.shape[-1]):
                    output.append("{:.2f}".format(pred[option]))
                output = "%s,%s" % (names[i, 0], ",".join(output))
                all_outputs.append(output)

            with open(os.path.join(script_dir, self.photo_dir, "outputs.csv"), "w+") as f:
                f.write("\n".join(all_outputs))

            with open(os.path.join(script_dir, self.photo_dir, "python_flag"), "w+") as f:
                f.write("")
            print("Textures have been classified in {:.3f} s".format(time() - start))
