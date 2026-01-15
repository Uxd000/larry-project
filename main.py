from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import os
import sys
import threading
import time

app = Flask(__name__)

# ---------------- CAMERA ----------------
cam = None
current_cat = "larry.jpeg"

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- THRESHOLDS ----------------
EYE_OPEN_THRESHOLD = 0.025
MOUTH_OPEN_THRESHOLD = 0.03
SQUINT_THRESHOLD = 0.018


# ---------------- CAT LOGIC ----------------
def cat_shock(lm):
    l_top, l_bot = lm.landmark[159], lm.landmark[145]
    r_top, r_bot = lm.landmark[386], lm.landmark[374]
    eye_open = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2
    return eye_open > EYE_OPEN_THRESHOLD


def cat_tongue(lm):
    return abs(lm.landmark[13].y - lm.landmark[14].y) > MOUTH_OPEN_THRESHOLD


def cat_glare(lm):
    l_top, l_bot = lm.landmark[159], lm.landmark[145]
    r_top, r_bot = lm.landmark[386], lm.landmark[374]
    squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2
    return squint < SQUINT_THRESHOLD


# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global cam, current_cat

    if cam is None:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            print("❌ Camera failed to open")
            return

    while True:
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        current_cat = "larry.jpeg"

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0]
            if cat_tongue(lm):
                current_cat = "cat-tongue.jpeg"
            elif cat_shock(lm):
                current_cat = "cat-shock.jpeg"
            elif cat_glare(lm):
                current_cat = "cat-glare2.jpeg"

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/cat")
def cat():
    return jsonify({"cat": current_cat})


@app.route("/stop", methods=["POST"])
def stop_program():
    shutdown_server()
    return "Stopped"


@app.route("/restart", methods=["POST"])
def restart_program():
    def delayed_restart():
        time.sleep(1)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    threading.Thread(target=delayed_restart).start()
    shutdown_server()
    return "Restarting"


# ---------------- HELPERS ----------------
def shutdown_server():
    global cam
    if cam is not None:
        cam.release()
        cam = None
    func = os.environ.get("WERKZEUG_SERVER_FD")
    os._exit(0)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
