import sqlite3

from seasonalweather.job_store import JobRepository

sqlite3.connect("forbidden.sqlite3")
_repository_type = JobRepository
