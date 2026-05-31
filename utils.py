"""
Shared utilities for the ISL pipeline.

Interview walkthrough order:
  1. mediapipe_detection  — BGR frame → Holistic landmarks
  2. landmarks_data       — landmarks → 150-D vector (pose + both hands)
  3. pad_sequence         — pad short videos to 30 frames for LSTM
  4. get_data / accuracy  — training & evaluation helpers
  5. prob_viz             — on-screen softmax bars during inference
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from scipy import stats

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Fixed feature sizes from MediaPipe Holistic (x,y only per landmark)
POSE_DIM = 33 * 2   # 66 — upper body pose context for signing
LH_DIM = 21 * 2     # 42 — left hand
RH_DIM = 21 * 2     # 42 — right hand
FEATURES_PER_FRAME = POSE_DIM + LH_DIM + RH_DIM  # 150


def mediapipe_detection(image, model):
    """Run MediaPipe Holistic on one OpenCV frame; returns annotated image + results."""
    if image is not None:
        # MediaPipe expects RGB; OpenCV captures BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # faster inference when buffer is read-only
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    else:
        return None, None


def draw_styled_landmarks(image, results):
    # Draw face connections
    if image is not None:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
    else:
        return


def landmarks_data(results):
    """
    Convert Holistic output to a single 150-dimensional feature vector.

    Layout: [pose (66) | left hand (42) | right hand (42)]
    Missing detections → zeros (avoids garbage when hand is off-screen / poor lighting).
  Face landmarks intentionally omitted — see README design section.
    """
    pose = np.array([[res.x, res.y] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_DIM)

    # Optional extension: 468 face landmarks × 3 (x,y,z) — not used for isolated greetings
    # face = np.array([[res.x, res.y, res.z] for res in
    #                  results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[res.x, res.y] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(LH_DIM)
    rh = np.array([[res.x, res.y] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(RH_DIM)

    return np.concatenate([pose, lh, rh])


def pad_sequence(data, max_frame_length):
    """Pad shorter videos with zero landmark frames so LSTM always receives shape (30, 150)."""
    padding = np.zeros(FEATURES_PER_FRAME)
    seq_length = len(data)

    if not seq_length == max_frame_length:
        diff = max_frame_length - seq_length
        for _ in range(diff):
            data.append(padding)

    return data


def accuracy(model, X_test, y_test):
    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    return accuracy_score(ytrue, yhat)


def get_data(train=False, test=True):
    sequences, labels = [], []
    data_folder = 'keypoint_data'

    actions = sorted(os.listdir(data_folder))
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:

        for video_file in os.listdir(os.path.join(data_folder, action)):
            data = np.load(os.path.join(data_folder, action, video_file))
            sequences.append(data)
            labels.append(label_map[action])

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    if train is False or test is True:
        return X_test, y_test
    elif train is True or test is False:
        return X_train, y_train


def prob_viz(res, actions, input_frame):
    """Overlay per-class softmax percentages on the webcam frame (demo / debugging)."""
    if res is not None:
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.putText(output_frame, f"{actions[num]} : {int(prob * 100)}% ", (10, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
        return output_frame

    else:
        output_frame = input_frame.copy()
        for num in range(len(actions)):
            prob = 0
            cv2.putText(output_frame, f"{actions[num]} : {int(prob * 100)}% ", (10, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
        return output_frame
