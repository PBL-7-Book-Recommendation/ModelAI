import os
import psycopg2
from psycopg2.extras import RealDictCursor


database_url = os.environ.get("DATABASE_URL")
print('database_url',database_url)

# Thông tin kết nối đến cơ sở dữ liệu
DB_HOST = os.environ.get("DB_HOST", "default_host")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_USER = os.environ.get("DB_USER", "default_user")
DB_PASS = os.environ.get("DB_PASS", "default_password")
DB_NAME = os.environ.get("DB_NAME", "default_database")

# Hàm kết nối đến cơ sở dữ liệu và thực hiện truy vấn
def query_db(query, params=None):
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, dbname=DB_NAME
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results