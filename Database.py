import sqlite3

class Database:
    def __init__(self):
        self.connection = sqlite3.connect('data.db')
        self.cursor = self.connection.cursor()

    def create_usertable(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

    def add_userdata(self, username, password):
        self.cursor.execute('INSERT INTO userstable VALUES (?,?)', (username, password))
        self.connection.commit()

    def login_user(self, username, password):
        self.cursor.execute('SELECT * FROM userstable WHERE username = ? AND password = ?', (username, password))
        data = self.cursor.fetchall()
        return data

    def is_admin(self, username, password):
        # Hardcoding the admin username and password for simplicity
        admin_username = "admin"
        admin_password = "admin_password"

        return username == admin_username and password == admin_password

    def view_all_users(self):
        self.cursor.execute('SELECT * FROM userstable')
        data = self.cursor.fetchall()
        return data
