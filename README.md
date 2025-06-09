# 🐍 Gesture-Controlled Snake Game

Control a classic Snake game using only your hand gestures via a webcam!  
This project combines **OpenCV**, **MediaPipe**, and **TensorFlow** to detect directional hand gestures and steer the snake in real time.

---

## 🎥 Demo Preview

> *(Optional)* Add a demo video or GIF here  
> Example:  
> ![Demo GIF](demo.gif)

---

## 🚀 Features

- 🎮 Real-time Snake game controlled by hand gestures  
- 🤚 Uses **MediaPipe** for hand landmark detection  
- 🧠 Trained with a custom neural network to classify gestures  
- 📷 Includes a tool for collecting your own gesture data  
- 📊 Visualizes training history and accuracy

---

## 📂 Project Structure

| File / Folder                | Description |
|-----------------------------|-------------|
| `collect_data.py`           | Script to collect gesture images from webcam |
| `train_model.py`            | Trains a neural network on the captured gesture data |
| `Snake_Game.py`             | Launches the real-time gesture-controlled Snake game |
| `gesture_landmark_model_best.h5` | Trained model used during gameplay |
| `gesture_data/` *(optional)*| Folder where gesture images are stored (one subfolder per gesture) |
| `training_history.png`      | Accuracy/loss plots of the training process |
| `Code_Report`               | Detailed explanation of the concepts used in the project |
| `README.md`                 | You're reading it! |

---

## 📦 Requirements

You can install the required dependencies using pip:

pip install opencv-python mediapipe tensorflow numpy matplotlib
