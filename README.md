# **AirTouch – Your Screen, Your Gestures, No Limits.**

## **📌 Project Overview**
**AirTouch** is a **gesture-based screen control system** that lets you **move your cursor, click, and double-click** using only your hand movements. No touchpad, no mouse – just your **webcam and gestures**! 🚀

## **🎯 Features**
✔ **Move the cursor** with your index finger.
✔ **Single Click** when your middle finger extends.
✔ **Double Click** when both middle and ring fingers extend.
✔ **Minimize Windows** with a fist gesture (optional).
✔ **Touchless & Hardware-Free** – No extra devices needed!

## **🛠 Technologies Used**
✅ **Python** – Main programming language  
✅ **MediaPipe** – Hand tracking & gesture recognition  
✅ **OpenCV** – Webcam feed processing  
✅ **PyAutoGUI** – Controlling the cursor & executing screen actions  
✅ **NumPy** – Efficient mathematical operations  

## **📌 Setup Instructions**
### **1️⃣ Clone the Repository**
```sh
 git clone https://github.com/YOUR_USERNAME/AirTouch.git
 cd AirTouch
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```
Or install them manually:
```sh
pip install opencv-python mediapipe pyautogui numpy
```

### **3️⃣ Run AirTouch**
```sh
python gesture_control.py
```

### **4️⃣ How to Use**
- **Move Cursor** → Keep your index finger extended.
- **Click** → Extend your middle finger while the index is straight.
- **Double Click** → Extend both middle & ring fingers.
- **Minimize Window (Optional)** → Make a fist.
- **Exit** → Press `'q'` in the OpenCV window.

## **📌 Troubleshooting & Tips**
- Ensure you have a **working webcam**.
- Run the script in a **well-lit environment** for better hand detection.
- If clicks happen unintentionally, adjust your **finger movements smoothly**.

## **📌 Contribution & License**
Feel free to contribute! Fork the repo, improve it, and submit a pull request.  
📝 Licensed under **MIT License**.

## **Ready to revolutionize touchless computing?** 

