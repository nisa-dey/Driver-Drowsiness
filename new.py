import streamlit.components.v1 as components
from secrets import choice
import streamlit as st
# Import necessary libraries
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import pygame
from datetime import datetime
import pandas as pd
from PIL import Image
import dlib
import Database
import time
st.markdown("""
<div class="container" style="background-color: #33FFFF; width: 800px; ">
    <nav class="navbar navbar-expand-lg bg-light" style="background-color: #33FFFF">
      <div class="container-fluid" style="background-color: #ffffff; border: 1px solid white; opacity: 0.6;">
        <a class="navbar-brand" href="index.php"></a> <button onclick="topFunction()" id="myBtn" class="myBtn" title="Go to top"><i style="color: black;" class="fa-solid fa-bars"></i></button>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav">
            <li class="nav-item">
              <a class="nav-link" style="color: Green; font-weight: bold; margin-right: 20px;"><img src="https://img.icons8.com/fluency-systems-filled/48/000000/about-us-male.png" style="width: 24px;"/>@--____________________ Detecting Drowsiness based on Camera Sensor _____________________--</a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
    </div>

""", unsafe_allow_html=True)

FRAME_WINDOW = st.image([])  # frame window

hhide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hhide_st_style, unsafe_allow_html=True)  # hide streamlit menu

db = Database.Database()


def main():

    col1, col2, col3 = st.columns(3)  # columns
    menu = ["HOME", "Signup", "Login", "Warnings"]  # menu
    choice = st.sidebar.selectbox("Menu", menu)  # sidebar menu

    pygame.mixer.init()

    if choice == "Home":
        st.subheader("Home")
    elif choice == "Login":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            db.create_usertable()
            result = db.login_user(username, password)
            if result:
                run = st.checkbox("START / STOP")  # checkbox
                if run == True:
                    pygame.mixer.init()
                    alarm_sound = pygame.mixer.Sound("D:\Detecting_Drowsiness_based_on_Camera_Sensor-main\warning.wav")
                    EYE_AR_THRESH = 0.3
                    EYE_AR_CONSEC_FRAMES = 21
                    MOUTH_AR_THRESH = 0.4
                    HEAD_TILT_ANGLE_THRESHOLD = 20
                    HEAD_TILT_DURATION_THRESHOLD = 3  # in seconds
                    ear = 0
                    mar = 0
                    X1 = []
                    X2 = []
                    face_cascade = cv2.CascadeClassifier("D:\Detecting_Drowsiness_based_on_Camera_Sensor-main\haarcascades\haarcascade_frontalface_default.xml")
                    COUNTER_FRAMES_EYE = 0
                    COUNTER_FRAMES_MOUTH = 0
                    COUNTER_BLINK = 0
                    COUNTER_MOUTH = 0
                    YAWN_COUNTER_THRESHOLD = 20
                    start_time = time.time()
                    tilt_start_time = None
                    look_forward_alarm = False
                    alarm_triggered = False

                    def eye_aspect_ratio(eye):
                        A = dist.euclidean(eye[1], eye[5])
                        B = dist.euclidean(eye[2], eye[4])
                        C = dist.euclidean(eye[0], eye[3])
                        return (A + B) / (2.0 * C)

                    def mouth_aspect_ratio(mouth):
                        A = dist.euclidean(mouth[5], mouth[8])
                        B = dist.euclidean(mouth[1], mouth[11])
                        C = dist.euclidean(mouth[0], mouth[6])
                        return (A + B) / (2.0 * C)

                    videoStream = cv2.VideoCapture(0)
                    ret, frame = videoStream.read()
                    size = frame.shape

                    detector = dlib.get_frontal_face_detector()
                    predictor = dlib.shape_predictor("D:\Detecting_Drowsiness_based_on_Camera_Sensor-main\shape_predictor_68_face_landmarks.dat")
                    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

                    model_points = np.array([(0.0, 0.0, 0.0),
                                            (0.0, -330.0, -65.0),
                                            (-225.0, 170.0, -135.0),
                                            (225.0, 170.0, -135.0),
                                            (-150.0, -150.0, -125.0),
                                            (150.0, -150.0, -125.0)])

                    focal_length = size[1]
                    center = (size[1] / 2, size[0] / 2)

                    camera_matrix = np.array([[focal_length, 0, center[0]],
                                            [0, focal_length, center[1]],
                                            [0, 0, 1]], dtype="double")

                    dist_coeffs = np.zeros((4, 1))

                    t_end = time.time()
                    while True:
                        ret, frame = videoStream.read()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        rects = detector(gray, 0)

                        face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

                        for (x, y, w, h) in face_rectangle:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            roi_gray = gray[y:y + h, x:x + w]
                            roi_color = frame[y:y + h, x:x + w]

                        for rect in rects:
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)
                            leftEye = shape[lStart:lEnd]
                            rightEye = shape[rStart:rEnd]
                            jaw = shape[48:61]

                            leftEAR = eye_aspect_ratio(leftEye)
                            rightEAR = eye_aspect_ratio(rightEye)
                            ear = (leftEAR + rightEAR) / 2.0
                            mar = mouth_aspect_ratio(jaw)
                            X1.append(ear)
                            X2.append(mar)
                            image_points = np.array([
                                (shape[30][0], shape[30][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                            ], dtype="double")

                            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                                        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                            translation_vector, camera_matrix, dist_coeffs)
                            for p in image_points:
                                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
                            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
                            proj_matrix = np.hstack((rvec_matrix, translation_vector))
                            euler_angles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
                            tilt_direction = "None"
                            if euler_angles[1] > HEAD_TILT_ANGLE_THRESHOLD:
                                tilt_direction = "Left"
                                if tilt_start_time is None:
                                    tilt_start_time = time.time()
                                else:
                                    tilt_duration = time.time() - tilt_start_time
                                    if tilt_duration >= HEAD_TILT_DURATION_THRESHOLD and not look_forward_alarm:
                                        if not alarm_triggered:
                                            alarm_sound.play()
                                            alarm_triggered = True
                                        with open('Warnings.csv', 'r+') as f:
                                            now = datetime.now()
                                            myDataList = f.readlines()
                                            dtString = now.strftime('%H:%M:%S')
                                            dStr = now.strftime('%d:%m:%Y')
                                            f.writelines(f'\n{username},{dStr},{dtString}')
                                        cv2.putText(frame, "Look forward!", (10, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        look_forward_alarm = True
                            elif euler_angles[1] < - HEAD_TILT_ANGLE_THRESHOLD:
                                tilt_direction = "Right"
                                if tilt_start_time is None:
                                    tilt_start_time = time.time()
                                else:
                                    tilt_duration = time.time() - tilt_start_time
                                    if tilt_duration >= HEAD_TILT_DURATION_THRESHOLD and not look_forward_alarm:
                                        if not alarm_triggered:
                                            alarm_sound.play()
                                            alarm_triggered = True
                                        with open('Warnings.csv', 'r+') as f:
                                            now = datetime.now()
                                            myDataList = f.readlines()
                                            dtString = now.strftime('%H:%M:%S')
                                            dStr = now.strftime('%d:%m:%Y')
                                            f.writelines(f'\n{username},{dStr},{dtString}')
                                        cv2.putText(frame, "Look forward!", (200, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        look_forward_alarm = True
                            else:
                                tilt_start_time = None
                                look_forward_alarm = False

                            cv2.putText(frame, f"Tilt Direction: {tilt_direction}", (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 0), 2)
                            p1 = (int(image_points[0][0]), int(image_points[0][1]))
                            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                            leftEyeHull = cv2.convexHull(leftEye)
                            rightEyeHull = cv2.convexHull(rightEye)
                            jawHull = cv2.convexHull(jaw)

                            cv2.drawContours(frame, [leftEyeHull], 0, (255, 255, 255), 1)
                            cv2.drawContours(frame, [rightEyeHull], 0, (255, 255, 255), 1)
                            cv2.drawContours(frame, [jawHull], 0, (255, 255, 255), 1)
                            cv2.line(frame, p1, p2, (255, 255, 255), 2)
                            if ear < EYE_AR_THRESH:
                                COUNTER_FRAMES_EYE += 1

                                if COUNTER_FRAMES_EYE >= EYE_AR_CONSEC_FRAMES:
                                    cv2.putText(frame, "Sleeping Driver!", (200, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    if not alarm_triggered:
                                        alarm_sound.play()
                                        alarm_triggered = True
                                    with open('Warnings.csv', 'r+') as f:
                                        now = datetime.now()
                                        myDataList = f.readlines()
                                        dtString = now.strftime('%H:%M:%S')
                                        dStr = now.strftime('%d:%m:%Y')
                                        f.writelines(f'\n{username},{dStr},{dtString}')
                            else:
                                if COUNTER_FRAMES_EYE > 2:
                                    COUNTER_BLINK += 1
                                COUNTER_FRAMES_EYE = 0

                            if mar >= MOUTH_AR_THRESH:
                                COUNTER_FRAMES_MOUTH += 1
                            else:
                                if COUNTER_FRAMES_MOUTH > 5:
                                    current_time = time.time()
                                    if current_time - start_time <= 30:
                                        COUNTER_MOUTH += 1
                                else:
                                    if COUNTER_MOUTH >= YAWN_COUNTER_THRESHOLD:
                                        if not alarm_triggered:
                                            alarm_sound.play()
                                            alarm_triggered = True
                                            with open('Warnings.csv', 'r+') as f:
                                                now = datetime.now()
                                                myDataList = f.readlines()
                                                dtString = now.strftime('%H:%M:%S')
                                                dStr = now.strftime('%d:%m:%Y')
                                                f.writelines(f'\n{username},{dStr},{dtString}')
                                            cv2.putText(frame, "YAWN ALERT!", (200, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    COUNTER_MOUTH = 0
                                    start_time = time.time()  
                                          
                                COUNTER_FRAMES_MOUTH = 0

                                if (time.time() - t_end) > 60:
                                    t_end = time.time()
                                    COUNTER_BLINK = 0

                        # Reset alarm trigger when condition is no longer met
                        if not (ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH):
                            alarm_triggered = False

                        cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, "Blinks: {}".format(COUNTER_BLINK), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Yawn count: {COUNTER_MOUTH:.2f}", (10, 90),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        FRAME_WINDOW.image(frame)
                        cv2.waitKey(1)
                else:
                    pass
            else:
                st.warning("Incorrect Username/Password")


    elif choice == "Signup":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        if st.button("Signup"):
            db.create_usertable()
            db.add_userdata(new_user, new_password)
            st.success("You have successfully created an valid Account")
            st.info("Go to Login Menu to login")

    # read data menu
    elif choice == 'Warnings':
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        login_button = st.sidebar.button("Login")
        if login_button:
            if db.is_admin(username, password):
                with col2:
                    st.write("Welcome, admin!")
                    df = pd.read_csv('Warnings.csv')
                    st.subheader("WARNINGS TIME AND DATE")
                    df = pd.read_csv('Warnings.csv')
                    st.write(df)
            else:
                st.warning("You do not have permission to access this page.Logged in as admin!")

    elif choice == 'HOME':
        st.markdown("""
        <div class="container" style="background-color: #33FFFF00; width: 800px; ">
        <nav class="navbar navbar-expand-lg bg-light" style="background-color: #33FFFF00">
          <div class="container-fluid" style="background-color: #33FFFF">
            <a class="navbar-brand" href="index.php"></a> <button onclick="topFunction()" id="myBtn" class="myBtn" title="Go to top"><i style="color: black;" class="fa-solid fa-bars"></i></button>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
              <ul class="navbar-nav">
                <li class="nav-item">
                  <a class="nav-link" style="color: Green; font-weight: bold; margin-right: 20px;">@--____________________ Detecting Drowsiness based on Camera Sensor _____________________--</a>
                </li>
              </ul>
            </div>
          </div>
        </nav>
        </div>

        """, unsafe_allow_html=True)
        with col1:
            st.image("DD.jpg", width=800, caption="Safety is always the choice of every wise, because they don't close eyes, so be wise.")


if __name__ == '__main__':
    main()