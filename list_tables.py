import sqlite3
conn = sqlite3.connect("data/northwind.sqlite")
cur = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
for name, typ in cur.fetchall():
    print(f"{typ}: {name}")
conn.close()
