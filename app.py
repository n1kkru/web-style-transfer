import io
import cv2
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from requests.auth import HTTPBasicAuth
from keras.models import model_from_json
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':

        f = request.files['file']
        f.save('img/before/' + secure_filename(f.filename))

        # Работа моей нейронной сети
        '''
        # opening and store file in a variable
        json_file = open('model/model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()

        # use Keras model_from_json to make a loaded model
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("model/model.h5")

        # compile and evaluate loaded model
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model = loaded_model


        IMG_HEIGHT, IMG_WIDTH = 128, 128

        image = cv2.imread('img/before/' + f.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Преобразование в нужный формат
        data = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        data[0] = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        # предсказание
        result = model.predict(data)

        pred = resize(result[0], (512, 512))
        pred = pred.reshape((1, 512, 512, 1))

        preds_test_thresh = (pred >= 0.5).astype(np.uint8)
        alpha_preds = preds_test_thresh * 255

        X_test_orig = np.zeros((1, 512, 512, 3), dtype=np.uint8)
        X_test_orig[0] = image

        predicted_masks = np.concatenate((X_test_orig, alpha_preds), axis=-1)

        mask = predicted_masks[0, :, :, :] # # can also write as preds_orig[1]
        # plt.imshow(mask)
        # mask.save('/Users/nikita/Downloads/im.png')
        Image.fromarray(mask).save('img/after/' + f.filename.split('.')[0] + '.png')
        '''
        # API
        file_path = 'img/before/' + secure_filename(f.filename)
        url = 'https://api.benzin.io/v1/removeBackground'
        auth = HTTPBasicAuth('X-Api-Key', '33e42363cb234d01b3269f62a7fa439a')
        files = {'image_file': open(f'{file_path}', 'rb')}

        req = requests.post(url, headers={'crop': 'True'}, auth=auth, files=files)

        # Сохранение в тот же файл
        # img = req.raw.read()
        # with open('2.jpg', 'wb') as f:
        #     f.write(req.content)

        image = Image.open(io.BytesIO(req.content))
        image.save('img/after/' + f.filename.split('.')[0] + '.png')
        Image.open(file_path).save('static/images/before.jpg')
        image.save('static/images/after.png')
        
        # –––––––––––––––––––– END PREDICT––––––––––––––––––––

    return render_template('prediction.html')


@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route('/login',  methods=['GET', 'POST'])
def login():
    return render_template('login.html')


@app.route('/signin',  methods=['GET', 'POST'])
def signin():
    return render_template('signin.html')


if __name__ == '__main__':
    app.run(debug=True, port=3000)
