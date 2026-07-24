import sqlite3


def open_job_database(path: str):
    return sqlite3.connect(path)
