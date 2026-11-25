"""DB 연결 풀. 앱이 뜰 때 열고 내려갈 때 닫는다."""
from psycopg_pool import ConnectionPool

from app import config

pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=4, open=False)
