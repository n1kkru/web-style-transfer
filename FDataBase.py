import sqlite3, math, time, re
from flask import url_for

class FDataBase:
    def __init__(self, db):
        self.__db = db
        self.__cur = db.cursor()

    # метод для добавления пользователя в БД
    def addUser(self, name, login, hpass):
        try:
            self.__cur.execute(f"SELECT COUNT() as 'count' FROM users WHERE login LIKE '{login}'")
            res = self.__cur.fetchone()
            if res['count'] > 0:
                print("Пользователь с таким login уже существует")
                return False
            tm = math.floor(time.time())
            self.__cur.execute("INSERT INTO 'users' VALUES(NULL, ?, ?, ?, ?)", (name, login, hpass, tm))
            self.__db.commit()
        except sqlite3.Error as e:
            print("Ошибка добавления пользователя в базу данных "+str(e))
            return False
        return True

    # метод для получения id пользователя из БД
    def getUser(self, user_id):
        try:
            self.__cur.execute(f"SELECT * FROM users WHERE id = {user_id} LIMIT 1")
            res = self.__cur.fetchone()
            if not res:
                print("Такой пользователь не найден!")
                return False
            return res
        except sqlite3.Error as e:
            print("Ошибка..." + str(e))

        return False

    def getUserByLogin(self, login):
        try:
            self.__cur.execute(f"SELECT * FROM users WHERE login = '{login}' LIMIT 1")
            res = self.__cur.fetchone()
            if not res:
                print("Такой пользователь не найден!")
                return False
            return res
        except sqlite3.Error as e:
            print("Ошибка..." + str(e))

        return False
