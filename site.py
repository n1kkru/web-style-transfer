import os
import cv2
import io
import sqlite3
import numpy as np
from PIL import Image
import requests
from requests.auth import HTTPBasicAuth
from FDataBase import FDataBase
from flask import Flask, render_template, url_for, session, redirect, request, abort, flash, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from UserLogin import UserLogin
from StyleTransfer import StyleTransfer

#конфигурация
DATABASE = '/tmp/site.db'
DEBUG = True
SECRET_KEY = 'dsgrbcbgn5d6554gfdgsdqqs,'

#создание приложения
app = Flask(__name__)
app.config.from_object(__name__)

#переопределение БД
app.config.update(dict(DATABASE=os.path.join(app.root_path, 'site.db')))

#ссылка на класс для управления авторизацией
login_manager = LoginManager(app)
#перенаправление на авторизацию
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return UserLogin().fromDB(user_id, dbase)

#функция для устоновления связи с БД
def connect_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

#создание БД без запуска веб-пирложения
def create_db():
    db = connect_db()
    with app.open_resource('sq_db.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    db.close()

# подключение в БД внутри запросов, если оно установлено
def get_db():
    if not hasattr(g, 'link_db'):
        g.link_db = connect_db()
    return g.link_db

create_db()

dbase = None
@app.before_request
def before_reqest():
    global dbase
    db = get_db()
    dbase = FDataBase(db)

@app.route("/", methods=["POST", "GET"])
@app.route("/home", methods=["POST", "GET"])
def index():
    # примеры обработки
    #example_before = "/static/images/before/1.jpg"
    #example_after = "/static/images/after/1_style.jpg"

    example_before = "/static/images/1_5.jpg"
    example_after = "/static/images/1400.png"
    STYLE = "static/images/STYLE.jpg"

    if request.method == 'POST':
        f = request.files['file']
        f.save('static/images/before/' + secure_filename(f.filename))
        file_path = 'static/images/before/' + secure_filename(f.filename)
        print('img dwnld')
        # API
        # file_path = 'img/before/' + secure_filename(f.filename)
        url = 'https://api.benzin.io/v1/removeBackground'
        auth = HTTPBasicAuth('X-Api-Key', '33e42363cb234d01b3269f62a7fa439a')
        files = {'image_file': open(f'{file_path}', 'rb')}

        req = requests.post(url, headers={'crop': 'True'}, auth=auth, files=files)
        print('ARI req')

        crop_image = Image.open(io.BytesIO(req.content))
        crop_image = crop_image.convert('RGB')
        # new_image = 'static/images/after/' + f.filename.split('.')[0] + '.png'
        new_image = 'static/images/after/crop.jpg'
        crop_image.save(new_image)
        print('img crop')
        a = StyleTransfer(new_image, STYLE)
        best = a.start()
        res = 'static/images/after/result.jpg'
        cv2.imwrite(res, best)

        # best.save()
        return render_template("home.html", title='Главная', before=file_path, after=res)

    return render_template("home.html", title='Главная', before=example_before, after=example_after)

@app.route("/info")
def info():
    return render_template("info.html", title='Информация', )

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        user = dbase.getUserByLogin(request.form['login'])
        if user and check_password_hash(user['pass'], request.form['pass']):
            userLogin = UserLogin().create(user)
            login_user(userLogin)
            return redirect(url_for('index'))
        else:
            flash("Логин/пароль введен неверно", "error")
    return render_template("login.html", title='Авторизация',)

@app.route("/me")
@login_required
def profile():
    u_id = current_user.get_id()
    return render_template("profile.html", title='Мой профиль', user=dbase.getUser(u_id))

@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == 'POST':
        if len(request.form['name']) > 4 and len(request.form['pass']) > 4 and request.form['pass'] == request.form['pass2']:
            hash = generate_password_hash(request.form['pass'])
            res = dbase.addUser(request.form['name'], request.form['login'], hash)
            if res:
                flash('Вы успешно зарегистрированы!')
                return redirect(url_for('login'))
            else:
                flash('Ошибка при добавлении в базу данных', "error")
        else:
            flash('Данные введены неверно!', "error")
    return render_template("register.html", title='Регистрация')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Вы вышли из аккаунта", "success")
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)