import pandas.io.sql
import pyarrow
import psycopg2
import json
import pandas as pd

conn = psycopg2.connect(
        host='localhost',
        database='walmart',
        user='postgres',
        password='123456lol'
    )

cur = conn.cursor()

product_query = 'SELECT p.title AS title, p.id AS id, p.category_id AS catid FROM product p'
cat_query = 'SELECT c.name AS name, c.id AS id, c.parent_id AS parent_id FROM category c'

df= pandas.io.sql.read_sql(cat_query, conn)
df.to_parquet('category_list.parquet', engine='pyarrow')
df.reset_index() 
df.to_json('hier.json', orient="split")

df = pandas.io.sql.read_sql(product_query,conn)
df.to_parquet('product_list.parquet', engine='pyarrow')
