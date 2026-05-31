"""
Real-time / video inference — sign language → on-screen text.

Flow: capture frame → MediaPipe → buffer 30 × 150-D vectors → LSTM predict
      → if max(softmax) > thresh, append label → draw sentence + probabilities.

For live webcam: change VideoCapture('Test_video.mp4') to VideoCapture(0).
Press 'q' to quit.
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from utils import mediapipe_detection, landmarks_data, prob_viz
from models import load_model


if __name__ == "__main__":
    sequence = []   # rolling buffer of landmark vectors (one per sampled frame)
    sentence = []   # recognized gesture labels shown on screen (max 5 unique in a row)
    predictions = []
    frame_count = 0
    res = None      # latest softmax output; None until first 30-frame window is full
    thresh = 0.85   # tuned on validation — reject uncertain predictions (implicit "unknown")

    mp_holistic = mp.solutions.holistic
    model = load_model('lstm_v3', pretrained=True, training=False)
    actions = os.listdir('greetings_data')
    cap = cv2.VideoCapture('Test_video.mp4')  # use 0 for default webcam
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()
            frame_count += 1

            if ret and frame_count % 2 == 0:
                image, results = mediapipe_detection(frame, holistic)
                # draw_styled_landmarks(image, results)  # optional skeleton overlay

                keypoints = landmarks_data(results)
                sequence.append(keypoints)

                # One LSTM inference per 30-frame window (~2 s of motion at skip-2 sampling)
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    sequence = []

                    # Only accept prediction above confidence threshold (reduces false positives)
                    if res[np.argmax(res)] > thresh:
                        predicted_label = actions[np.argmax(res)]
                        # Avoid repeating the same label back-to-back
                        if len(sentence) > 0:
                            if predicted_label != sentence[-1]:
                                sentence.append(predicted_label)
                        else:
                            sentence.append(predicted_label)

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                image = prob_viz(res, actions, image)

                cv2.rectangle(image, (0, 0), (width, 40), (0, 0, 0), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            elif ret is False:
                break
