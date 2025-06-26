<p align="center">
  <a href="https://play.google.com/store/apps/dev?id=7086930298279250852" target="_blank">
    <img alt="" src="https://github-production-user-asset-6210df.s3.amazonaws.com/125717930/246971879-8ce757c3-90dc-438d-807f-3f3d29ddc064.png" width=500/>
  </a>  
</p>

### Our facial recognition algorithm is globally top-ranked by NIST in the FRVT 1:1 leaderboards. <span><img src="https://github.com/kby-ai/.github/assets/125717930/bcf351c5-8b7a-496e-a8f9-c236eb8ad59e" alt="badge" width="36" height="20"></span>  
[Latest NIST FRVT evaluation report 2024-12-20](https://pages.nist.gov/frvt/html/frvt11.html)  

![FRVT Sheet](https://github.com/user-attachments/assets/16b4cee2-3a91-453f-94e0-9e81262393d7)

#### 🆔 ID Document Liveness Detection - Linux - [Here](https://web.kby-ai.com)  <span><img src="https://github.com/kby-ai/.github/assets/125717930/bcf351c5-8b7a-496e-a8f9-c236eb8ad59e" alt="badge" width="36" height="20"></span>
#### 🤗 Hugging Face - [Here](https://huggingface.co/kby-ai)
#### 📚 Product & Resources - [Here](https://github.com/kby-ai/Product)
#### 🛟 Help Center - [Here](https://docs.kby-ai.com)
#### 💼 KYC Verification Demo - [Here](https://github.com/kby-ai/KYC-Verification-Demo-Android)


# FaceSearch-Docker
## Overview
This repository demonstrates `1:N face recognition`, `face search SDK` derived from `KBY-AI`'s [face recognition server SDK](https://hub.docker.com/r/kbyai/face-recognition) by implementing the functionalities to register face  and search face from database(`PostgreSQL`).<br/>
This repo offers APIs to enroll face, to search face, to see database, to clear database. And every API can be customized by updating [app.py](https://github.com/kby-ai/FaceSearch-Docker/blob/main/app.py) file accordingly.</br>

> In this repo, we integrated `KBY-AI`'s face recognition solution into `Linux Server SDK` by docker container.<br/>
> We can customize the SDK to align with customer's specific requirements.

### ◾FaceSDK(Server) Product List
  | No.      | Repository | SDK Details |
  |------------------|------------------|------------------|
  | 1        | [Face Liveness Detection - Linux](https://github.com/kby-ai/FaceLivenessDetection-Docker)    | Face Livness Detection |
  | 2        | [Face Liveness Detection - Windows](https://github.com/kby-ai/FaceLivenessDetection-Windows)    | Face Livness Detection |
  | 3        | [Face Liveness Detection - C#](https://github.com/kby-ai/FaceLivenessDetection-CSharp-.Net)    | <b>Face Livness Detection |
  | 4        | [Face Recognition - Linux](https://github.com/kby-ai/FaceRecognition-Docker)    | Face Recognition |
  | 5        | [Face Recognition - Windows](https://github.com/kby-ai/FaceRecognition-Windows)    | Face Recognition |
  | 6        | [Face Recognition - C#](https://github.com/kby-ai/FaceRecognition-CSharp-.NET)    | Face Recognition |
  | ➡️        | <b>[Face Search - Linux](https://github.com/kby-ai/FaceSearch-Docker)</b>    | <b>Face Search</b> |

> To get Face SDK(mobile), please visit products [here](https://github.com/kby-ai/Product):<br/>

## Try the API
### Postman Endpoints
  To test the `API`, you can use `Postman`. Here are the [endpoints](https://github.com/kby-ai/FaceSearch-Docker/blob/main/kby-ai-facesearch.postman_collection.json) for testing:
1. `http://<your-base-url>/register`</br>
  This `API` enrolls face data from image base64 format and save it to database(`PostgreSQL`)</br>
2. `http://<your-base-url>/search`</br>
  This `API` seeks face similar to input face among database and returns enrolled image ID and similarity score.</br>
3. `http://<your-base-url>/user_list`</br>
  This `API` shows all data enrolled on database(`PostgreSQL`).</br>
4. `http://<your-base-url>/remove_all`</br>
  This `API` removes all data from database.<br>

  ![image](https://github.com/user-attachments/assets/ca670eb8-3f86-468b-a7ba-d6ab6dedffaf)

## SDK License

`Face Search SDK` requires a license per machine.</br>
- The code below shows how to use the license: https://github.com/kby-ai/FaceSearch-Docker/blob/fbfe7dd40972fe3e5e4b1e82a604155263b7623b/app.py#L33-L45

- To request the license, please provide us with the `machine code` obtained from the `getMachineCode` function.

#### Please contact us:</br>
🧙`Email:` contact@kby-ai.com</br>
🧙`Telegram:` [@kbyai](https://t.me/kbyai)</br>
🧙`WhatsApp:` [+19092802609](https://wa.me/+19092802609)</br>
🧙`Discord:` [KBY-AI](https://discord.gg/CgHtWQ3k9T)</br>
🧙`Teams:` [KBY-AI](https://teams.live.com/l/invite/FBAYGB1-IlXkuQM3AY)</br>
  
## How to run

### 1. System Requirements
  - `CPU`: `2` cores or more (Recommended: `2` cores)
  - `RAM`: `4GB` or more (Recommended: `8GB`)
  - `HDD`: `4GB` or more (Recommended: `8GB`)
  - `OS`: `Ubuntu 20.04` or later
  - Dependency: `OpenVINO™` Runtime (Version: `2022.3`)

### 2. Setup and Test
  - Clone the project:
    ```bash
    git clone https://github.com/kby-ai/FaceSearch-Docker.git
    ```
  - Download the model from Google Drive: [click here](https://drive.google.com/file/d/1ExXnc-QMVCFtGoP3xOkjoQFq56hO0PV0/view?usp=sharing)
    ```bash
    cd FaceSearch-Docker
    
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19vA7ZOlo19BcW8v4iCoCGahUEbgKCo48' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19vA7ZOlo19BcW8v4iCoCGahUEbgKCo48" -O data.zip && rm -rf /tmp/cookies.txt
    
    unzip data.zip
    ```
  - Build the `Docker` image:
    ```bash
    sudo docker build --pull --rm -f Dockerfile -t kby-ai-face-search:latest .
    ```
  - Get `machine code`
    ```bash
    sudo docker run -e LICENSE="xxxxx" kby-ai-face-search:latest
    ```
  - Send us the `machine code` obtained.
    ![image](https://github.com/user-attachments/assets/cb3590a3-2f68-4e68-8ae2-21522a222abc)
  - Update the `license.txt` file by overwriting the license key that you received from `KBY-AI` team.
  - Run the `Docker` container:
    ```bash
    sudo docker run -v ./license.txt:/root/kby-ai-face/license.txt -p 8081:8080 -p 9001:9000 kby-ai-face-search:latest
    ```
    ![image](https://github.com/user-attachments/assets/dd8e2ba2-2121-43e0-b8d9-fa0c2b10dc70)

## About SDK

### 1. Initializing the SDK

- Step One

  First, obtain the `machine code` for activation and request a license based on the `machine code`.
  ```python
  machineCode = getMachineCode()
  print("machineCode: ", machineCode.decode('utf-8'))
  ```
  
- Step Two

  Next, activate the SDK using the received license.
  ```python
  setActivation(license.encode('utf-8'))
  ```  
  If activation is successful, the return value will be `SDK_SUCCESS`. Otherwise, an error value will be returned.

- Step Three

  After activation, call the initialization function of the SDK.
  ```python
  initSDK("data".encode('utf-8'))
  ```
  The first parameter is the path to the model.

  If initialization is successful, the return value will be `SDK_SUCCESS`. Otherwise, an error value will be returned.

### 2. Enum and Structure
  - SDK_ERROR
  
    This enumeration represents the return value of the `initSDK` and `setActivation` functions.

    | Feature| Value | Name |
    |------------------|------------------|------------------|
    | Successful activation or initialization        | 0    | SDK_SUCCESS |
    | License key error        | -1    | SDK_LICENSE_KEY_ERROR |
    | AppID error (Not used in Server SDK)       | -2    | SDK_LICENSE_APPID_ERROR |
    | License expiration        | -3    | SDK_LICENSE_EXPIRED |
    | Not activated      | -4    | SDK_NO_ACTIVATED |
    | Failed to initialize SDK       | -5    | SDK_INIT_ERROR |

- FaceBox
  
    This structure represents the output of the face detection function.

    | Feature| Type | Name |
    |------------------|------------------|------------------|
    | Face rectangle        | int    | x1, y1, x2, y2 |
    | Face angles (-45 ~ 45)        | float    | yaw, roll, pitch |
    | Face quality (0 ~ 1)        | float    | face_quality |
    | Face luminance (0 ~ 255)       | float    | face_luminance |
    | Eye distance (pixels)       | float    | eye_dist |
    | Eye closure (0 ~ 1)       | float    | left_eye_closed, right_eye_closed |
    | Face occlusion (0 ~ 1)       | float    | face_occlusion |
    | Mouth opening (0 ~ 1)       | float    | mouth_opened |
    | 68 points facial landmark        | float [68 * 2]    | landmarks_68 |
    | Face templates        | unsigned char [2048]    | templates |

### 3. Main Functions
  - Face Detection
  
    The `Face SDK` provides a single API for detecting faces, determining `face orientation` (yaw, roll, pitch), assessing `face quality`, detecting `facial occlusion`, `eye closure`, `mouth opening`, and identifying `facial landmarks`.
    
    The function can be used as follows:

    ```python
    faceBoxes = (FaceBox * maxFaceCount)()
    faceCount = faceDetection(image_np, image_np.shape[1], image_np.shape[0], faceBoxes, maxFaceCount)
    ```
    
    This function requires 5 parameters.
    * The first parameter: the byte array of the RGB image buffer.
    * The second parameter: the width of the image.
    * The third parameter: the height of the image.
    * The fourth parameter: the `FaceBox` array allocated with `maxFaceCount` for storing the detected faces.
    * The fifth parameter: the count allocated for the maximum `FaceBox` objects.

    The function returns the count of the detected face.

  - Create Template

    The SDK provides a function that enables the generation of `template`s from RGB data. These `template`s can be used for face verification between two faces.

    The function can be used as follows:

    ```python    
    templateExtraction(image_np1, image_np1.shape[1], image_np1.shape[0], faceBoxes1[0])
    ```

    This function requires 4 parameters.
    * The first parameter: the byte array of the RGB image buffer.
    * The second parameter: the width of the image.
    * The third parameter: the height of the image.
    * The fourth parameter: the `FaceBox` object obtained from the `faceDetection` function.

    If the `template` extraction is successful, the function will return `0`. Otherwise, it will return `-1`.
    
  - Calculation similiarity

    The `similarityCalculation` function takes a byte array of two `template`s as a parameter. 

    ```python
    similarity = similarityCalculation(faceBoxes1[0].templates, faceBoxes2[0].templates)
    ```

    It returns the similarity value between the two `template`s, which can be used to determine the level of likeness between the two individuals.

### 4. Thresholds
  The default thresholds are as the following below:
  https://github.com/kby-ai/FaceSearch-Docker/blob/df84c977b2f78d82c9c3b630f0e6ae5f1885ca57/app.py#L22-L24

