import sys
sys.path.append('.')

import os
import numpy as np
import base64
import io
import FaceManage.manage as db_manage
import time
from datetime import datetime
import uuid

from PIL import Image
from flask import Flask, request, jsonify
from facesdk import getMachineCode
from facesdk import setActivation
from facesdk import initSDK
from facesdk import faceDetection
from facesdk import templateExtraction
from facesdk import similarityCalculation
from facebox import FaceBox

verifyThreshold = 0.67
maxFaceCount = 15

licensePath = "license.txt"
license = ""

# Get a specific environment variable by name
license = os.environ.get("LICENSE")

# Check if the variable exists
if license is not None:
    print("Value of LICENSE:")
else:
    license = ""
    try:
        with open(licensePath, 'r') as file:
            license = file.read().strip()
    except IOError as exc:
        print("failed to open license.txt: ", exc.errno)
    print("license: ", license)

machineCode = getMachineCode()
print("machineCode: ", machineCode.decode('utf-8'))

ret = setActivation(license.encode('utf-8'))
print("activation: ", ret)

ret = initSDK("data".encode('utf-8'))
print("init: ", ret)

app = Flask(__name__) 

db_manage.open_database()

def generate_unique_image_name():
    """
    Generate a unique name for an image file.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")  # Current timestamp to microseconds
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    unique_name = f"{timestamp}_{unique_id}"
    return unique_name

def image_to_blob(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')  # You can use other formats like 'PNG'
    return img_byte_arr.getvalue()

def blob_to_image(blob_data):
    return Image.open(io.BytesIO(blob_data))

@app.route('/compare_face', methods=['POST'])
def compare_face():
    file1 = request.files['file1']
    file2 = request.files['file2']

    try:
        image1 = Image.open(file1).convert('RGB')
    except:
        result = "Failed to open file1"
        response = jsonify({"resultCode": result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


    try:
        image2 = Image.open(file2).convert('RGB')
    except:
        result = "Failed to open file2"
        response = jsonify({"resultCode": result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    image_np1 = np.asarray(image1)
    image_np2 = np.asarray(image2)

    faceBoxes1 = (FaceBox * maxFaceCount)()
    faceCount1 = faceDetection(image_np1, image_np1.shape[1], image_np1.shape[0], faceBoxes1, maxFaceCount)

    faceBoxes2 = (FaceBox * maxFaceCount)()
    faceCount2 = faceDetection(image_np2, image_np2.shape[1], image_np2.shape[0], faceBoxes2, maxFaceCount)

    faces1_result = []
    faces2_result = []
    for i in range(faceCount1):
        templateExtraction(image_np1, image_np1.shape[1], image_np1.shape[0], faceBoxes1[i])

        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes1[i].landmark_68[j * 2], "y": faceBoxes1[i].landmark_68[j * 2 + 1]})

        face = {"x1": faceBoxes1[i].x1, "y1": faceBoxes1[i].y1, "x2": faceBoxes1[i].x2, "y2": faceBoxes1[i].y2, 
                      "yaw": faceBoxes1[i].yaw, "roll": faceBoxes1[i].roll, "pitch": faceBoxes1[i].pitch,
                      "face_quality": faceBoxes1[i].face_quality, "face_luminance": faceBoxes1[i].face_luminance, "eye_dist": faceBoxes1[i].eye_dist,
                      "left_eye_closed": faceBoxes1[i].left_eye_closed, "right_eye_closed": faceBoxes1[i].right_eye_closed,
                      "face_occlusion": faceBoxes1[i].face_occlusion, "mouth_opened": faceBoxes1[i].mouth_opened,
                      "landmark_68": landmark_68}
        
        faces1_result.append(face)

    for i in range(faceCount2):
        templateExtraction(image_np2, image_np2.shape[1], image_np2.shape[0], faceBoxes2[i])

        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes2[i].landmark_68[j * 2], "y": faceBoxes2[i].landmark_68[j * 2 + 1]})


        face = {"x1": faceBoxes2[i].x1, "y1": faceBoxes2[i].y1, "x2": faceBoxes2[i].x2, "y2": faceBoxes2[i].y2, 
                      "yaw": faceBoxes2[i].yaw, "roll": faceBoxes2[i].roll, "pitch": faceBoxes2[i].pitch,
                      "face_quality": faceBoxes2[i].face_quality, "face_luminance": faceBoxes2[i].face_luminance, "eye_dist": faceBoxes2[i].eye_dist,
                      "left_eye_closed": faceBoxes2[i].left_eye_closed, "right_eye_closed": faceBoxes2[i].right_eye_closed,
                      "face_occlusion": faceBoxes2[i].face_occlusion, "mouth_opened": faceBoxes2[i].mouth_opened,
                      "landmark_68": landmark_68}
        
        faces2_result.append(face)

    
    if faceCount1 > 0 and faceCount2 > 0:
        results = []
        for i in range(faceCount1):
            for j in range(faceCount2): 
                similarity = similarityCalculation(faceBoxes1[i].templates, faceBoxes2[j].templates)
                match_result = {"face1": i, "face2": j, "similarity": similarity}
                results.append(match_result)

        response = jsonify({"resultCode": "Ok", "faces1": faces1_result, "faces2": faces2_result, "results": results})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    elif faceCount1 == 0:
        response = jsonify({"resultCode": "No face1", "faces1": faces1_result, "faces2": faces2_result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    elif faceCount2 == 0:
        response = jsonify({"resultCode": "No face2", "faces1": faces1_result, "faces2": faces2_result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

@app.route('/compare_face_base64', methods=['POST'])
def compare_face_base64():
    content = request.get_json()

    try:
        imageBase64_1 = content['base64_1']
        image_data1 = base64.b64decode(imageBase64_1)    
        image1 = Image.open(io.BytesIO(image_data1)).convert('RGB')
    except:
        result = "Failed to open file1"
        response = jsonify({"resultCode": result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    
    try:
        imageBase64_2 = content['base64_2']
        image_data2 = base64.b64decode(imageBase64_2)
        image2 = Image.open(io.BytesIO(image_data2)).convert('RGB')
    except IOError as exc:
        result = "Failed to open file1"
        response = jsonify({"resultCode": result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    image_np1 = np.asarray(image1)
    image_np2 = np.asarray(image2)

    faceBoxes1 = (FaceBox * maxFaceCount)()
    faceCount1 = faceDetection(image_np1, image_np1.shape[1], image_np1.shape[0], faceBoxes1, maxFaceCount)

    faceBoxes2 = (FaceBox * maxFaceCount)()
    faceCount2 = faceDetection(image_np2, image_np2.shape[1], image_np2.shape[0], faceBoxes2, maxFaceCount)

    faces1_result = []
    faces2_result = []
    for i in range(faceCount1):
        templateExtraction(image_np1, image_np1.shape[1], image_np1.shape[0], faceBoxes1[i])

        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes1[i].landmark_68[j * 2], "y": faceBoxes1[i].landmark_68[j * 2 + 1]})

        face = {"x1": faceBoxes1[i].x1, "y1": faceBoxes1[i].y1, "x2": faceBoxes1[i].x2, "y2": faceBoxes1[i].y2, 
                      "yaw": faceBoxes1[i].yaw, "roll": faceBoxes1[i].roll, "pitch": faceBoxes1[i].pitch,
                      "face_quality": faceBoxes1[i].face_quality, "face_luminance": faceBoxes1[i].face_luminance, "eye_dist": faceBoxes1[i].eye_dist,
                      "left_eye_closed": faceBoxes1[i].left_eye_closed, "right_eye_closed": faceBoxes1[i].right_eye_closed,
                      "face_occlusion": faceBoxes1[i].face_occlusion, "mouth_opened": faceBoxes1[i].mouth_opened,
                      "landmark_68": landmark_68}
        
        faces1_result.append(face)

    for i in range(faceCount2):
        templateExtraction(image_np2, image_np2.shape[1], image_np2.shape[0], faceBoxes2[i])

        landmark_68 = []
        for j in range(68):
            landmark_68.append({"x": faceBoxes2[i].landmark_68[j * 2], "y": faceBoxes2[i].landmark_68[j * 2 + 1]})


        face = {"x1": faceBoxes2[i].x1, "y1": faceBoxes2[i].y1, "x2": faceBoxes2[i].x2, "y2": faceBoxes2[i].y2, 
                      "yaw": faceBoxes2[i].yaw, "roll": faceBoxes2[i].roll, "pitch": faceBoxes2[i].pitch,
                      "face_quality": faceBoxes2[i].face_quality, "face_luminance": faceBoxes2[i].face_luminance, "eye_dist": faceBoxes2[i].eye_dist,
                      "left_eye_closed": faceBoxes2[i].left_eye_closed, "right_eye_closed": faceBoxes2[i].right_eye_closed,
                      "face_occlusion": faceBoxes2[i].face_occlusion, "mouth_opened": faceBoxes2[i].mouth_opened,
                      "landmark_68": landmark_68}
        
        faces2_result.append(face)

    
    if faceCount1 > 0 and faceCount2 > 0:
        results = []
        for i in range(faceCount1):
            for j in range(faceCount2): 
                similarity = similarityCalculation(faceBoxes1[i].templates, faceBoxes2[j].templates)
                match_result = {"face1": i, "face2": j, "similarity": similarity}
                results.append(match_result)

        response = jsonify({"resultCode": "Ok", "faces1": faces1_result, "faces2": faces2_result, "results": results})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    elif faceCount1 == 0:
        response = jsonify({"resultCode": "No face1", "faces1": faces1_result, "faces2": faces2_result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    elif faceCount2 == 0:
        response = jsonify({"resultCode": "No face2", "faces1": faces1_result, "faces2": faces2_result})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

@app.route("/register", methods=['POST'])
def enroll_user():
    faceCount = 0
    content = request.get_json()

    try:
        imageBase64 = content['base64Image']
        image_data = base64.b64decode(imageBase64)
        image = Image.open(io.BytesIO(image_data))
    except:
        result = "Invalid image data"
        response = jsonify({"status": "error", "message": result})
        response.status_code = 400
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    image_np = np.asarray(image)

    faceBoxes = (FaceBox * maxFaceCount)()
    faceCount = faceDetection(image_np, image_np.shape[1], image_np.shape[0], faceBoxes, maxFaceCount)
    faceref_id_list = []

    SiteID, country, source_image_id = content['collection_id'], "Brazil", content['person_id']
    
    if db_manage.check_db(source_image_id) == -1:
        result = 'Duplicate person_id'
        response = jsonify({"status": "error", "message": result})
        response.status_code = 409
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    
    if faceCount > 0:
        for i in range(faceCount):
            templateExtraction(image_np, image_np.shape[1], image_np.shape[0], faceBoxes[i])

            capture_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            faceref_id = generate_unique_image_name()

            _ = db_manage.register_face(faceref_id, faceBoxes[i].templates, capture_time, SiteID, country, source_image_id)
            result = 'success'
            faceref_id_list.append(faceref_id)
    else:
        result = 'No face detected'
        response = jsonify({"status": "error", "message": result, "person_id": source_image_id, "faceID": faceref_id_list})
        response.status_code = 422
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    response = jsonify({"status": result, "person_id": source_image_id, "faceID": faceref_id_list})
    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route("/remove_all", methods=['POST'])
def remove_all():
    db_manage.clear_database()

    result = 'All users removed'
    response = jsonify({"status": "success", "message": result})
    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route("/user_list", methods=['POST'])
def user_list():
    userlist = db_manage.get_userlist()

    response = jsonify({"status": "success", "users": userlist})
    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route("/remove_user", methods=['POST'])
def remove_user():
    content = request.get_json()
    name, SiteID = content['face_id'], content['collection_id']

    db_manage.remove_user(name, SiteID)
    response = jsonify({"status": "success"})
    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route("/search", methods=['POST'])
def verify_user():
    faceCount = 0

    content = request.get_json()
    try:
        imageBase64 = content['base64Image']
        image_data = base64.b64decode(imageBase64)
        image = Image.open(io.BytesIO(image_data))
    except:
        result = "Invalid image data"
        response = jsonify({"status": "error", "message": result})

        response.status_code = 400
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    image_np = np.asarray(image)

    faceBoxes = (FaceBox * maxFaceCount)()
    faceCount = faceDetection(image_np, image_np.shape[1], image_np.shape[0], faceBoxes, maxFaceCount)
    
    SiteID, StartDateTime, EndDateTime, ConfidenceThreshold, Threshold = content['collection_id'], None, None, float(content['similarity']), None

    result = 'Verify Failed'
    name = ''
    face_score = 0

    if faceCount == 1:
        templateExtraction(image_np, image_np.shape[1], image_np.shape[0], faceBoxes[0])
        id, fname, face_score, fsource_image_id = db_manage.search_faces(faceBoxes[0].templates, ConfidenceThreshold)
        if id >= 0:
            result, name, imageids = 'success', fname, fsource_image_id
        if id == -2:
            result = 'Database empty. Please register first.'
            response = jsonify({"status": "error", "message": result})
            response.status_code = 404
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            return response
        if id == -1:
            result = 'No matching user found'
            response = jsonify({"status": "error", "message": result})
            response.status_code = 404
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            return response

    elif faceCount > 1:
        result = 'Multiple faces detected'
        response = jsonify({"status": "error", "message": result})
        response.status_code = 422
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    else:
        result = 'No face detected'
        response = jsonify({"status": "error", "message": result})
        response.status_code = 422
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    
    response = jsonify({"status": result, "person_id": imageids, "similarity_detected": str(face_score), "face_id": name})
    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response
  
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
