# FLask API that interacts with the UI and handles all the backend work

import smooth_tiled_predictions
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from predict_here import *
import predict_here as pr
import cv2
from tensorflow import keras
from flask import Flask
from flask import render_template, request
import os
from PIL import Image
import PIL
import sys
sys.path.append('WebApp')

app = Flask(__name__)
UPLOAD_FOLDER = 'static\images'

# Model Work

# Loading the model
model = keras.models.load_model('mobilenet.h5',
                                custom_objects={
                                    "dice_coef": dice_coef,
                                    "dice_coef_loss": dice_coef_loss
                                })


# Reacting to the 'segment' button on the UI
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_files = request.files['image']
        if image_files:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_files.filename
            )
            image_files.save(image_location)
            y = cv2.imread(image_location)

            pred = predict_with_image(y, model)
            output_image = os.path.join(
                UPLOAD_FOLDER,
                'output',
                image_files.filename
            )
            # print(output_image)
            # plt.imshow(pred)
            pred.save(output_image)
            # output_image = os.path.join(
            #     "static", "output", "image.png"
            # )
            return render_template("index.html", image_in=image_location, image=output_image)
    return render_template("index.html", image_in=None, image=None)

    return render_template("index.html", prediction=0)
    # We can send some variables to the index html file using prediction variable.


if __name__ == "__main__":
    app.run(port=12000, debug=True)
