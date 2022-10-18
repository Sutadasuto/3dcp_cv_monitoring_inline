import os
import warnings
from time import sleep, time

from texture.data import calculate_features
from joblib import load
from texture.utils import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import user parameters from config file
from texture import config

config_dict = config.__dict__  # Parameters for feature extraction
photo_dir = "images"


class TextureClassifier():
    def __init__(self):
        self.config_dict = config_dict
        self.photo_dir = photo_dir

        # Load preliminary variables
        self.script_dir = os.path.split(os.path.realpath(__file__))[0]
        self.scaler = load(os.path.join(self.script_dir, "data_scaler.joblib"))
        self.enc = load(os.path.join(self.script_dir, "class_encoder.joblib"))
        self.class_names = self.enc.categories_[0]
        self.clf = None
        try:
            self.best_feats = load(os.path.join(self.script_dir, "best_features.joblib"))
        except FileNotFoundError:
            self.best_feats = None
            warnings.warn("There is no file with to read the list of best features. Using all features for prediction.")

    def matlab_flag_exists(self):
        if os.path.exists(os.path.join(self.script_dir, self.photo_dir, "matlab_flag")):
            return True
        return False

    def classify_textures(self):
        print("Waiting to classify textures...")
        while not self.matlab_flag_exists():
            sleep(0.1)
        print("Textures have been received.")

        start = time()
        X, names, feature_names = calculate_features(os.path.join(self.script_dir, self.photo_dir), **self.config_dict)
        print("Features extracted in {:.3f} s".format(time() - start))

        if len(X) == 0:
            with open(os.path.join(self.script_dir, self.photo_dir, "python_flag"), "w+") as f:
                f.write("")
            warnings.warn("There were no images to classify. Finishing the process without predictions")
            return None

        # Data pre-processing as done during training
        X = self.scaler.transform(X)

        # Select the best features used for training (if available)
        if self.best_feats is not None:
            X = X[:, self.best_feats]

        # Load classifier (if it is not yet ready)
        if self.clf is None:
            self.clf = create_model(input_size=X.shape[-1], output_size=len(self.class_names), conv_size=6, dropout=0.3)
            self.clf.load_weights(os.path.join(self.script_dir, "weights.hdf5"))

        start = time()
        preds = self.clf.predict(X[:, None, :, None])
        all_outputs = ["file," + ",".join(self.class_names.tolist())]

        for i, pred in enumerate(preds):
            output = []
            for option in range(pred.shape[-1]):
                output.append("{:.2f}".format(pred[option]))
            output = "%s,%s" % (names[i, 0], ",".join(output))
            all_outputs.append(output)

        with open(os.path.join(self.script_dir, self.photo_dir, "outputs.csv"), "w+") as f:
            f.write("\n".join(all_outputs))

        with open(os.path.join(self.script_dir, self.photo_dir, "python_flag"), "w+") as f:
            f.write("")
        print("Textures have been classified in {:.3f} s".format(time() - start))
