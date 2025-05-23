import psycopg2
import sys
import os
import numpy as np
import ctypes

from facesdk import similarityCalculation

database_base_name = 'users'
table_name = "feature"
postgres_insert_blob_query = "INSERT INTO " + table_name + " (id, name, features, capture_time, country, source_image_id, SiteID) VALUES (%s, %s, %s, %s, %s, %s, %s)"
postgres_create_table_query = """
CREATE TABLE IF NOT EXISTS """ + table_name + """ (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    features BYTEA NOT NULL,
    capture_time TEXT NOT NULL,
    country TEXT NOT NULL,
    source_image_id TEXT NOT NULL,
    SiteID TEXT NOT NULL
)
"""
postgres_update_all_query = "UPDATE " + table_name + " SET name = %s, features = %s, capture_time = %s, country = %s, source_image_id = %s, SiteID = %s WHERE id = %s"
postgres_search_query = "SELECT * FROM " + table_name
postgres_delete_all = "DELETE FROM " + table_name
postgres_delete_user = "DELETE FROM " + table_name + " WHERE name = %s"

data_all = []
MATCHING_THRES = 0.67
FEATURE_SIZE = 2048
max_id = -1

face_database = None

# Database connection details
db_config = {
    "host": "imagexlabs-db.cr52lfhn77fz.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "dbname": "Imagex_fr",
    "user": "imagexfr_usr",
    "password": "imagexfr@5632!@#",
    "sslmode": "prefer",
    "connect_timeout": 10
}

# Open database
def open_database():
    global max_id
    global face_database

    try:
        face_database = psycopg2.connect(**db_config)
        print("Connected to PostgreSQL database")
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return

    try:
        with face_database.cursor() as cursor:
            # Create table if not exists
            cursor.execute(postgres_create_table_query)
            face_database.commit()

            # Load data from "feature" table
            cursor.execute(postgres_search_query)
            rows = cursor.fetchall()

            for row in rows:
                id = row[0]
                name = row[1]
                features = np.frombuffer(row[2], dtype=np.uint8)
                capture_time = row[3]
                SiteID = row[6]
                country = row[4]
                source_image_id = row[5]

                if not features.shape[0] == FEATURE_SIZE:
                    continue

                features = features.reshape(1, FEATURE_SIZE)
                data_all.append({'id': id, 'name': name, 'features': features, 'capture_time': capture_time, 'country': country, 'source_image_id': source_image_id, 'SiteID': SiteID})
                if id > max_id:
                    max_id = id

        print('>>>>>>>>>>>> Load Users', len(data_all))
    except psycopg2.Error as e:
        print(f"Error during database operations: {e}")
    finally:
        if face_database:
            face_database.commit()

# Clear the PostgreSQL database
def clear_database():
    global face_database
    data_all.clear()
    cursor = face_database.cursor()
    cursor.execute(postgres_delete_all)
    face_database.commit()
    cursor.close()

# Check whether db has same source_image_id
def check_db(image_id):
    if len(data_all) == 0:
        return -2

    for data in data_all:
        find_source_image_id = data['source_image_id']

        if find_source_image_id == image_id:
            return -1          
    return 0

# Register a new face
def register_face(name, features, capture_time, SiteID, country, source_image_id):
    global face_database, max_id
    max_id += 1
    # Convert ctypes array to NumPy array
    if isinstance(features, ctypes.Array):
        features = np.ctypeslib.as_array(features)
    cursor = face_database.cursor()
    cursor.execute(postgres_insert_blob_query, (max_id, name, features.tobytes(), capture_time, country, source_image_id, SiteID))
    face_database.commit()
    cursor.close()

    data_all.append({'id': max_id, 'name': name, 'features': features, 'capture_time': capture_time, 'country': country, 'source_image_id': source_image_id, 'SiteID': SiteID})
    return max_id

# Update an existing face
def update_face(id=None, name=None, features=None, user_face=None):
    global face_database
    cursor = face_database.cursor()
    cursor.execute(postgres_update_all_query, (name, features.tobytes(), user_face, id))
    face_database.commit()
    cursor.close()

# Verify a face
def verify_face(feat):
    max_score = 0

    if len(data_all) == 0:
        return -2, None, None, None

    find_id, find_name = -1, None
    for data in data_all:
        id = data['id']
        features = data['features']
        # Convert NumPy array back to ctypes array
        features_ctypes = (ctypes.c_ubyte * FEATURE_SIZE)(*features.flatten())
        score = similarityCalculation(feat, features_ctypes)

        if score >= max_score:
            max_score = score
            find_id = id
            find_name = data['name']
            find_source_image_id = data['source_image_id']

    if max_score >= MATCHING_THRES:
        return find_id, find_name, max_score, find_source_image_id

    return -1, None, None, None

def search_faces(feat, ConfidenceThreshold):
    if len(data_all) == 0:
        return -2, None, None, None

    find_name_list, score_list, find_source_image_id_list = [], [], []
    sorted_name_list, sorted_score_list, sorted_source_image_id_list = [], [], []
    for data in data_all:
        features = data['features']
        # Convert NumPy array back to ctypes array
        features_ctypes = (ctypes.c_ubyte * FEATURE_SIZE)(*features.flatten())
        score = similarityCalculation(feat, features_ctypes)
        if score >= ConfidenceThreshold:
            find_name_list.append(data['name'])
            score_list.append(score)
            find_source_image_id_list.append(data['source_image_id'])

    sorted_score_index = np.argsort(score_list)
    sorted_score_index = np.flipud(sorted_score_index)
    for i in range(len(score_list)):
        sorted_score_list.append(score_list[sorted_score_index[i]])
        sorted_name_list.append(find_name_list[sorted_score_index[i]])
        sorted_source_image_id_list.append(find_source_image_id_list[sorted_score_index[i]])
    if len(sorted_score_list) == 0:
        return -1, None, None, None
    else:
        return 0, sorted_name_list[0], sorted_score_list, sorted_source_image_id_list

# Get user list
def get_userlist():
    return [(data['id'], data['source_image_id'], data['SiteID']) for data in data_all]

# Remove a user
def remove_user(name):
    global face_database, data_all
    cursor = face_database.cursor()
    cursor.execute(postgres_delete_user, (name,))
    face_database.commit()
    cursor.close()

    data_all = [i for i in data_all if not (i['name'] == name)]

