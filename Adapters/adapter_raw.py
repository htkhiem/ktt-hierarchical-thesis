import pandas.io.sql
import pyarrow
import psycopg2
import json
import pandas as pd
import json

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

cat = pd.read_parquet('category_list.parquet', engine='pyarrow')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(cat)
    
from collections import namedtuple

# Hierarchical relationships are to be externally recorded for simplicity.
Node = namedtuple('Node', ['name', 'db_id','parent_id'])

# Build the tree. Root node has no name and  ID = -1 to completely avoid
# collision risks
root_node = Node('', -1,None)

# In the view, parent-less categories have NaN for parent_id. We can replace
# them with -1 (the special internal ID we use for root_note above).
# This avoids an if-statement and helps with straight-line execution performance.
category_sql_table = cat
category_sql_table['parent_id'].fillna(-1, inplace=True)
category_sql_table = category_sql_table.reset_index()
# We must use some sort of storage as we cannot be guaranteed to be able to
# iterate through the table in a hierarchical manner.
# As such, this node dict allows for easy access to all nodes at once, no
# matter the iteration order.
node_dict = {
    -1: root_node
}
# Auxillary dict to keep track of child nodes. This is stored externally as
# we would often reach a child node before the parent node is discovered.
children_dict = {
    -1: []
}

for row in category_sql_table.itertuples():
    node = Node(row[2], row[3], row[4])
    node_dict[row[3]] = node
    try:
        children_dict[row[4]].append(node)
    except KeyError:
        children_dict[row[4]] = [node]
        
# We now have a full tree, accessible through root_node.
from queue import Queue

even_queue = Queue()
odd_queue = Queue()

try:
    for child in children_dict[root_node.db_id]:
        odd_queue.put(child)
except KeyError:
        # Node has no children
        print('Root has no children')


depth = 1  # which queue to pop depends on depth
# parent_of = []
put_queue = even_queue
pop_queue = odd_queue
classes = []
cls2idx = {}
level_offsets = [0]
level_sizes = []

while not even_queue.empty() or not odd_queue.empty():
    level_sizes.append(pop_queue.qsize())
    
    while not pop_queue.empty():
        node = pop_queue.get()
        idx = len(classes)
        classes.append(node.name)
        cls2idx[node.name] = idx
        try:
            for child in children_dict[node.db_id]:
                put_queue.put(child)
        except KeyError:
            # Node has no children
            pass
    # Flip queues
    level_offsets.append([*cls2idx.values()][-1]+1) #
    temp = pop_queue
    pop_queue = put_queue
    put_queue = temp
    depth += 1
    

# Build a mapping from database ID to internal indices
id2idx = {}
for db_id,node in node_dict.items():
    try:
        id2idx[db_id] = cls2idx[node.name]
    except:
        pass
    
#new loop to create parent_of
try:
    for child in children_dict[root_node.db_id]:
        odd_queue.put(child)
except KeyError:
        # Node has no children
        print('Root has no child')


depth = 1  # which queue to pop depends on depth
parent_of = []
put_queue = even_queue
pop_queue = odd_queue

while not even_queue.empty() or not odd_queue.empty():
    temp_list = []
    
    while not pop_queue.empty():
        node = pop_queue.get()
        try:
            temp_list.append(id2idx[node.parent_id] - level_offsets[depth-2]) #?
        except:
            temp_list.append(id2idx[node.db_id])
        try:
            for child in children_dict[node.db_id]:
                put_queue.put(child)
        except KeyError:
            # Node has no children
            pass
    # Flip queues

    parent_of.append(temp_list)
    temp = pop_queue
    pop_queue = put_queue
    put_queue = temp
    depth += 1
    
print('id2idx mapping & parent_of mapping test')
print('ID 82 in DB:', node_dict[82].name)
print('Maps to')
print('Index {} in internal representing {}'.format(id2idx[82],classes[id2idx[82]]))
print('Level offsets', level_offsets[1-2])
print('Its parent is {} with index of'.format(classes[parent_of[1][id2idx[82] - level_offsets[1-2]]]), parent_of[1][id2idx[82]-level_offsets[1-2]]) 

data_set = {"classes": classes, 
            "level_offsets": level_offsets,
            "level_sizes": level_sizes,
            "parent_of": parent_of
            }
with open('test.json', 'w') as json_file:
    json.dump(data_set, json_file)