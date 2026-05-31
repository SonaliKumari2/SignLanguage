# Interview Guide — Real-Time ISL Translation

Use this document to prepare for technical, ML, and system-design questions about this project. Each section lists **likely questions** and **concise talking points** aligned with the actual codebase.

---

## 1. Project Overview & Motivation

**Q: What does your project do?**  
A: Real-time translation of **isolated Indian Sign Language greeting gestures** into text using a webcam/video, OpenCV, MediaPipe Holistic, and a stacked LSTM classifier.

**Q: What problem does it solve?**  
A: It helps bridge communication between hearing-impaired signers and non-signers by showing recognized gesture labels on screen in real time.

**Q: Is this continuous sign language recognition?**  
A: No. It is **isolated gesture classification** (one sign ≈ one label). Continuous SLR is an active research area with scarce datasets.

**Q: What gestures can it recognize?**  
A: Nine greeting classes in `greetings_data/`: alright, good afternoon, good evening, good morning, good night, hello, how are you, pleased, thank you.

**Q: What is the novelty of your approach?**  
A: End-to-end lightweight pipeline (no RGB CNN on pixels), holistic body+hands features, cached `.npy` preprocessing for fast iteration, confidence-gated predictions, and multiple architecture variants (LSTM v1–v3, Transformer stub).

---

## 2. MediaPipe & Computer Vision

**Q: What are the 21 hand landmarks MediaPipe extracts?**  
A: One wrist + four points per finger (thumb, index, middle, ring, pinky) = 21 points. Each has (x, y, z); this project uses **x, y only** (normalized image coordinates).

**Q: Why 150 features per frame and not 63 (21×3)?**  
A: We use **Holistic**, not hand-only: 33 pose × 2 + 21 left hand × 2 + 21 right hand × 2 = **150**. Hand-only 21×3=63 is a different, narrower feature set.

**Q: What if MediaPipe fails to detect hands or pose?**  
A: `landmarks_data()` fills missing pose/left/right blocks with **zeros** so the LSTM always receives a fixed-size vector—avoiding garbage coordinates.

**Q: Why not use face landmarks?**  
A: Face adds ~1400+ dimensions and compute; for **isolated greetings**, body+hands suffice. Face helps questions vs statements in full sentence-level SLU.

**Q: What are `min_detection_confidence` and `min_tracking_confidence`?**  
A: MediaPipe Holistic thresholds (0.5 in code). Low confidence → missed landmarks → zero padding.

**Q: Why OpenCV?**  
A: Cross-platform video I/O (webcam/files), display, and drawing overlays for predictions.

**Q: Why process every 2nd frame (`frame_count % 2`)?**  
A: Reduces redundant adjacent frames and CPU load while keeping enough temporal resolution for a 30-frame buffer.

---

## 3. Data & Preprocessing

**Q: Describe your dataset.**  
A: Multiple `.MOV` samples per class under `greetings_data/<gesture>/`, recorded performers doing isolated signs.

**Q: Why save `.npy` files?**  
A: NumPy stores `(30, 150)` arrays efficiently; training skips repeated MediaPipe passes (hours → minutes).

**Q: Why 30 frames per sample?**  
A: At ~30 FPS with skip-2 sampling, 30 frames ≈ ~2 s of motion—enough for short isolated greetings with fixed LSTM input shape.

**Q: What if a video has fewer than 30 frames?**  
A: `pad_sequence()` appends zero landmark vectors until length is 30.

**Q: What if a video has more than 30 frames?**  
A: Extraction stops at 30 processed samples (`processed == MAX_FRAME_LENGTH`).

**Q: How would you handle variable signing speed in production?**  
A: Options discussed in README: longer fixed window, **padded variable length** with masking, **sliding windows**, **confidence counter** (predict after N consecutive high-confidence windows), or **gesture boundary detection**.

**Q: Train/test split?**  
A: `train.py`: 90/10 via `train_test_split`. `evaluate.py` / `get_data()`: 95/5 when loading test data (be aware of slight inconsistency if asked).

---

## 4. Machine Learning & LSTM

**Q: Why LSTM and not CNN or a simple neural network?**  
A: Gestures are **sequences over time**. CNNs on single frames miss temporal dynamics; LSTMs maintain hidden state across frames.

**Q: Explain LSTM gates.**  
A: **Forget gate** — what to drop from cell state; **input gate** — what to store; **output gate** — what to expose; **cell state** — long-term memory of the gesture trajectory.

**Q: What is your input tensor shape?**  
A: `(batch, 30, 150)` — 30 timesteps, 150 features per timestep.

**Q: Describe your model architecture.**  
A: Default `lstm_v3`: three LSTM layers (64→256→128 units) then Dense 1024→512→128→64→softmax over 9 classes.

**Q: Why softmax on the last layer?**  
A: Multi-class classification; outputs sum to 1.0 (probability distribution over gestures).

**Q: What loss and optimizer?**  
A: `categorical_crossentropy` + Adam (lr 3e-4) — standard for one-hot multi-class labels.

**Q: What is the 0.85 threshold?**  
A: Hyperparameter tuned on validation: only show prediction if `max(softmax) > 0.85` to reduce false positives (“unknown” otherwise).

**Q: Why not Transformer for this project?**  
A: Transformer is implemented (`models.py`) but with **small data** and **short sequences**, LSTM is simpler and less likely to overfit. Transformers shine with large datasets and long-range dependencies.

**Q: Difference between LSTM and Transformer for sequences?**  
A: LSTM processes sequentially (information passes through each timestep); Transformer self-attention connects distant frames in one hop but needs more data/parameters.

**Q: What callbacks did you use?**  
A: `ModelCheckpoint` (best weights), `ReduceLROnPlateau` (lower LR when val_loss plateaus).

**Q: How do you prevent duplicate predictions on screen?**  
A: Only append if the new argmax label differs from `sentence[-1]`; keep last 5 labels max.

---

## 5. Evaluation & Metrics

**Q: How do you evaluate the model?**  
A: `evaluate.py` loads `isl_model.h5`, predicts on hold-out `.npy` sequences, reports **accuracy** via `sklearn.metrics.accuracy_score`.

**Q: Explain precision, recall, and F1.**  
A: **Precision** — of predicted “hello”, how many were correct; **Recall** — of all true “hello”, how many detected; **F1** — balance of both.

**Q: What is a confusion matrix?**  
A: Table of true vs predicted classes; shows which gestures are confused (e.g., “good morning” vs “good afternoon”).

**Q: Reported accuracies?**  
A: See `models/lstm_v1/info.txt` and `models/lstm_v3/info.txt` — e.g. lstm_v3 ~91.5% train, ~80% val (your run may vary).

**Q: Why is validation accuracy lower than training?**  
A: Possible overfitting, limited data per class, or subject/video diversity in the split.

---

## 6. Real-Time Inference & Engineering

**Q: Walk through the inference pipeline.**  
A: Read frame → MediaPipe → `landmarks_data()` → append to list → at 30 frames, `model.predict` → if confidence > threshold, update sentence → `prob_viz` + `putText` → `imshow`.

**Q: How would you switch from test video to webcam?**  
A: `cv2.VideoCapture(0)` instead of `'Test_video.mp4'`.

**Q: What is `prob_viz`?**  
A: Overlays per-class softmax percentages on the frame for debugging and demos.

**Q: Why save model as `.h5`?**  
A: HDF5 stores Keras weights/architecture for reuse without retraining. Modern alternative: `.keras` format.

**Q: How would you deploy to mobile/edge?**  
A: Convert to TFLite, quantize, run MediaPipe on-device, reduce model size, optimize frame rate.

**Q: Bottlenecks in real time?**  
A: MediaPipe holistic per frame, LSTM predict every 30 frames (acceptable); drawing and capture on CPU.

---

## 7. Two Hands & Feature Engineering (Extensions)

**Q: One hand or two hands?**  
A: **Both** — left and right hand landmarks (42 values) plus pose (66).

**Q: Did you add cross-hand features?**  
A: Not in current code; interview extension: relative distance, angle between wrists, symmetry, dominant hand — pad zeros if one hand missing.

**Q: Would you use z (depth)?**  
A: MediaPipe provides z; this project uses x,y only for smaller vectors and sufficient 2D-normalized discrimination for greetings.

---

## 8. Limitations & Ethics

**Q: Main limitations?**  
A: Closed vocabulary, isolated signs, fixed window, lighting/pose sensitivity, small dataset, no grammar/sentence formation.

**Q: Bias and fairness concerns?**  
A: Model may fail for different skin tones, backgrounds, camera angles, or signers not in training data — need diverse data and user testing.

**Q: False positives vs false negatives trade-off?**  
A: Higher threshold (0.85) favors fewer false labels but may miss valid signs (more false negatives).

---

## 9. Code Walkthrough Questions

**Q: Where are landmarks converted to numbers?**  
A: `utils.landmarks_data()`.

**Q: Where is padding implemented?**  
A: `utils.pad_sequence()` in `keypoint_extraction.py`.

**Q: Where are models defined?**  
A: `models.py` — `lstm_v1`, `lstm_v2`, `lstm_v3`, `transformer`, `load_model()`.

**Q: Which model does `main.py` use?**  
A: `load_model('lstm_v3', pretrained=True, training=False)` — loads `.h5` from `models/lstm_v3/`.

**Q: What does `train.py` use?**  
A: `load_model('lstm_v1', pretrained=False)` — trains from scratch (changeable).

---

## 10. System Design & “What If” Questions

**Q: How would you scale to 500 gesture classes?**  
A: More data, hierarchical classification, contrastive pretraining on landmarks, regularization, maybe Transformer + masking with more compute.

**Q: How would you build continuous sentence recognition?**  
A: Segment signs (boundary detection), sequence-to-sequence model, larger corpus, language model for word ordering.

**Q: How would you add speech output?**  
A: TTS (gTTS/pyttsx3) on recognized text after threshold check.

**Q: How would you improve robustness to signing speed?**  
A: Dynamic time warping on sequences, adaptive window, or consecutive high-confidence frame counter before predicting.

**Q: Compare fixed sliding window vs confidence counter approach.**  
A: Fixed window can cut slow signs or add noise for fast signers; confidence counter waits until N consecutive confident frames before firing prediction.

---

## 11. Quick-Fire Technical Terms

| Term | One-line answer |
|------|-----------------|
| ISL | Indian Sign Language |
| Holistic | MediaPipe solution combining pose + hands (+ face available) |
| Isolated sign | Single sign performed alone, not in continuous discourse |
| One-hot encoding | `to_categorical` for class labels |
| Return sequences | LSTM outputs per timestep (True) vs single vector (False) |
| Vanishing gradient | Why LSTM replaced vanilla RNN for long sequences |
| Categorical crossentropy | Loss for multi-class one-hot targets |

---

## 12. Questions You Can Ask the Interviewer

- Does the team work on **edge deployment** or cloud-only inference?  
- Is there an existing **sign language dataset** standard internally?  
- How do they evaluate **fairness** across user populations?  

---

*Tip: Open `main.py`, `utils.py`, and `models.py` side-by-side while answering—comments in those files map directly to this guide.*
