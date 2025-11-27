import sqlite3
conn = sqlite3.connect('data/northwind.sqlite')
for stmt in [
    "CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;",
    "CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;",
    "CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;"
]:
    try:
        conn.execute(stmt)
        print(f"Success: {stmt}")
    except Exception as e:
        print(f"Error: {stmt} -> {e}")
conn.commit()
conn.close()