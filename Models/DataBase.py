import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class PostgresqlDB:
    def __init__(self, user="selectel", password="selectel",
                 host="selectel-pgdocker", port="5432", database="selectel"):
        self.connection = psycopg2.connect(user=user,
                                      # пароль, который указали при установке PostgreSQL
                                      password=password,
                                      host=host,
                                      port=port,
                                      database=database)
        self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.connection.cursor()

    def get_on_id(self, id):
        self.cursor = self.connection.cursor()
        sql_create_database = f'select * from wiki where id = {id};'
        self.cursor.execute(sql_create_database)
        record = self.cursor.fetchone()

        return record