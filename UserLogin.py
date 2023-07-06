class UserLogin():
    # функция для наполнения класса
    def fromDB(self, user_id, db):
        self.__user = db.getUser(user_id)
        return self
    def create(self, user):
        self.__user = user
        return self

    # функция проверки авторизации
    def is_authenticated(self):
        return True

    # функция проверки активности
    def is_active(self):
        return True

    # функция определяющая неавторизованных пользователей
    def is_anonymous(self):
        return False

    # функция для получения id текущего пользователя
    def get_id(self):
        return str(self.__user['id'])
