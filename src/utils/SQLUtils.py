import sqlite3


class SQLiteBase:

    def __init__(self, path):
        self.path = path
        self.__db_connection = sqlite3.connect(path)
        self.cur = self.__db_connection.cursor()

    def close(self):
        self.__db_connection.close()

    def execute(self, new_data):
        self.cur.execute(new_data)

    def fetchone(self, sql):
        self.execute(sql)
        result = self.cur.fetchone()
        return result

    def fetchall(self, sql):
        self.execute(sql)
        result = self.cur.fetchall()
        return result

    def create_table(self, name, collist):
        txt = "CREATE TABLE IF NOT EXISTS {0}(".format(name)
        i = 0
        for col in collist:
            if i == 0:
                txt += col
            else:
                txt += ',' + col
        txt += ')'
        self.cur.execute(txt)

    def commit(self):
        self.__db_connection.commit()

    def __del__(self):
        self.__db_connection.close()

    def __enter__(self):
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        self.cur.close()
        if isinstance(exc_value, Exception):
            self.__db_connection.rollback()
        else:
            self.__db_connection.commit()
        self.__db_connection.close()


class OptunaDB(SQLiteBase):

    def __init__(self, path):
        super(OptunaDB, self).__init__(path)

    def get_best_trial(self):
        """returns trial number - 1 so you can go into the correct directory
        (directory names start with 0, trial indices in sqlite start with 1)"""

        results = self.fetchall("select trial_id, study_id, value from trials WHERE value IS NOT NULL order by value asc limit 10")
        print("Top 10 trials: \n")
        for r in results:
            print("{0}: {1}\n".format(r[0], r[2]))
        return results[0][0]-1

