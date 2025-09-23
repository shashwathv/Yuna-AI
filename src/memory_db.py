import psycopg2
from psycopg2.extras import RealDictCursor

class YunaMemoryDB:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="yuna_memory",
            user="guts",
            password="",
            host="localhost",
            port=5432
        )
        self.conn.autocommit = True

    def save_message(self, user_id, session_id, role, message):
        with self.conn.cursor()as cur:
            cur.execute(
                "INSERT INTO conversations(user_id, session_id, role, message) VALUES(%s, %s, %s, %s)",
             (user_id, session_id, role, message)
            )
    
    def get_recent_messages(self, user_id, limit=10):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, message FROM conversations WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
                (user_id, limit)
            )
            return cur.fetchall()