"""SQLite helper: find DB, introspect schema, run queries and return columns/rows/errors.
"""
import os
import sqlite3
from typing import List, Tuple, Optional, Any

DEFAULT_CANDIDATES = [
    os.path.join(os.getcwd(), "data", "northwind.sqlite"),
    os.path.join(os.getcwd(), "data", "northwind.db"),
    os.path.join(os.getcwd(), "data", "northwind.sqlite" , "northwind.db"),
]


def find_db_path(candidates: Optional[List[str]] = None) -> Optional[str]:
    if candidates is None:
        candidates = DEFAULT_CANDIDATES
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class SQLiteTool:
    def __init__(self, path: Optional[str] = None):
        self.db_path = path or find_db_path()
        if not self.db_path:
            raise FileNotFoundError("Could not find northwind sqlite DB in data/; tried candidates.")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def get_tables(self) -> List[str]:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        return [r[0] for r in cur.fetchall()]

    def get_columns(self, table: str) -> List[str]:
        cur = self.conn.execute(f"PRAGMA table_info('{table}')")
        return [r[1] for r in cur.fetchall()]

    def run(self, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]], Optional[str]]:
        try:
            cur = self.conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [tuple(r) for r in cur.fetchall()]
            return cols, rows, None
        except Exception as e:
            return [], [], str(e)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
