from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER ='results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULTS_FOLDER  

kernel = 'static/colorize/pts_in_hull.npy'

def colorize_image(img_path):
    net = cv2.dnn.readNetFromCaffe("static/colorize/colorization_deploy_v2.prototxt",
                                   "static/colorize/colorization_release_v2.caffemodel")
    pts_in_hull = np.load(kernel)

    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    img = cv2.imread(img_path)
    img_rgb = (img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)

    img_l = img_lab[:, :, 0]
    (H_orig, W_orig) = img_rgb.shape[:2]
    img_rs = cv2.resize(img_rgb, (224, 224))
    img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:, :, 0]
    img_l_rs -= 50
    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward('class8_ab')[0, :, :, :].transpose((1, 2, 0))
    (H_out, W_out) = ab_dec.shape[:2]
    ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
    img_bgr_out = cv2.cvtColor(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2RGB)
    final_img='static/final_img.png'
    rgb_image = cv2.cvtColor(img_bgr_out, cv2.COLOR_BGR2RGB)
    cv2.imwrite(final_img, (rgb_image * 255).astype(np.uint8))
    return final_img

@app.route('/')
def index():
    return render_template('index.html', final_mask=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        main_image_file = request.files['file']
        main_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(main_image_file.filename))
        main_image_file.save(main_image_path)
        
        colorized_img = colorize_image(main_image_path)
        
        return jsonify(final_req_image=colorized_img)

if __name__ == '__main__':
    app.run(debug=True)