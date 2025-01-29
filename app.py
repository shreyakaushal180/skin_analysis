from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
from collections import deque

app = Flask(__name__)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

skin_tone_history = deque(maxlen=20)  # Store the last 20 skin tone values

def calculate_average_rgb(region):
    """Calculate the average RGB value of the given region."""
    if region.size == 0:
        return (0, 0, 0)  # Return default RGB value for empty regions
    avg_color_per_row = np.mean(region, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return tuple(map(int, avg_color)) if avg_color.size == 3 else (0, 0, 0)

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame.")
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                face_region = frame[y:y+h, x:x+w]
                skin_rgb = calculate_average_rgb(face_region)
                skin_tone_history.append(skin_rgb)

                # Calculate the weighted average skin tone from the history
                weights = np.linspace(1, 2, len(skin_tone_history))
                weighted_avg_skin_tone = np.average(skin_tone_history, axis=0, weights=weights).astype(int)
                skin_tone_text = f'Skin Tone: R={weighted_avg_skin_tone[2]}, G={weighted_avg_skin_tone[1]}, B={weighted_avg_skin_tone[0]}'
                cv2.putText(frame, skin_tone_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame.")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/skin_analysis')
def skin_analysis():
    if len(skin_tone_history) == 0:
        return redirect(url_for('index'))

    # Calculate the final average skin tone
    final_avg_skin_tone = np.mean(skin_tone_history, axis=0).astype(int)
    skin_tone = {
        'R': final_avg_skin_tone[2],
        'G': final_avg_skin_tone[1],
        'B': final_avg_skin_tone[0]
    }

    # Determine skin type and provide skin care tips (simplified example)
    skin_type = "Normal"  # Placeholder for actual skin type determination logic
    skin_care_tips = "Stay hydrated and use sunscreen daily."  # Placeholder for actual tips

    return render_template('skin_analysis.html', skin_tone=skin_tone, skin_type=skin_type, skin_care_tips=skin_care_tips)

if __name__ == '__main__':
    app.run(debug=True)