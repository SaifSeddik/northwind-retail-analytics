CREATE VIEW IF NOT EXISTS lowercase_products AS
SELECT *
FROM Products;

CREATE VIEW IF NOT EXISTS lowercase_customers AS
SELECT *
FROM Customers;

CREATE VIEW IF NOT EXISTS lowercase_orders AS
SELECT *
FROM Orders;

CREATE VIEW IF NOT EXISTS lowercase_orderdetails AS
SELECT *
FROM "Order Details";