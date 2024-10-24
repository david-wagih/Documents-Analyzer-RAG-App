import sqlite3
import json


def create_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn

def create_table(conn):
    sql_create_documents_table = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        embedding TEXT NOT NULL
    );
    """
    conn.execute(sql_create_documents_table)
    conn.commit()

def insert_document(conn, doc_id, embedding):
    sql = ''' INSERT INTO documents(id, embedding)
              VALUES(?,?) '''
    embedding_json = json.dumps(embedding)  # Convert embedding to JSON string
    cur = conn.cursor()
    cur.execute(sql, (doc_id, embedding_json))
    conn.commit()
    return cur.lastrowid

def fetch_all_documents(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents")
    rows = cur.fetchall()
    return [(row[0], json.loads(row[1])) for row in rows]  # Convert JSON string back to list
