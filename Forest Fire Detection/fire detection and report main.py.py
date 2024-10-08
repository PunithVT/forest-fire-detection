import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

# Load the trained model
model = load_model('C:/Users/punit/PycharmProjects/fire_detection_model.h5')

# Email configuration (Use environment variables for security)
sender_email = os.getenv('SENDER_EMAIL', '.......@outlook.com')
sender_password = os.getenv('SENDER_PASSWORD', '........')
receiver_email = os.getenv('RECEIVER_EMAIL', '.........@gmail.com')
smtp_server = "smtp-mail.outlook.com"
smtp_port = 587

# Function to send an email alert
def send_email_alert():
    try:
        # Create the email content
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "Fire Detection Alert"
        body = "Fire has been detected at Your Location1! Take necessary action immediately."
        msg.attach(MIMEText(body, 'plain'))
        
        # Set up the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Send the email
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        st.success("Email alert sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email alert: {str(e)}")

# Function to preprocess the frame

def preprocess_frame(frame):
    frame = cv2.resize(frame, (150, 150))  # Match the size used during training
    frame = frame / 255.0  # Normalize the frame
    frame = np.expand_dims(frame, axis=0)  # Reshape for model input (1, 150, 150, 3)
    return frame

# Function to predict fire in a frame
def predict_fire(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    return prediction[0][0] < 0.45

# Streamlit app
st.title("Forest Fire Detection")

# Option selection
option = st.selectbox("Select input type", ("Upload Image", "Upload Video", "Use Live Webcam"))

if option == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)

        # Perform fire detection
        if predict_fire(frame):
            st.write("Fire Detected")
            st.image(frame, caption='Fire Detected', use_column_width=True)
            send_email_alert()  # Send email alert
        else:
            st.write("No Fire")
            st.image(frame, caption='No Fire', use_column_width=True)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        fire_detected = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform fire detection
            if predict_fire(frame):
                label = "Fire Detected"
                color = (0, 0, 255)  # Red color for fire
                if not fire_detected:
                    send_email_alert()  # Send email alert once
                    fire_detected = True
            else:
                label = "No Fire"
                color = (0, 255, 0)  # Green color for no fire

            # Display the label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Convert the frame to RGB (from BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            stframe.image(frame, channels="RGB")
        
        cap.release()

elif option == "Use Live Webcam":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        # Place the checkbox outside the loop
        stop_button_pressed = st.checkbox("Stop Webcam", key="stop_button")

        stframe = st.empty()
        fire_detected = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame.")
                break

            # Perform fire detection
            if predict_fire(frame):
                label = "Fire Detected"
                color = (0, 0, 255)  # Red color for fire
                if not fire_detected:
                    send_email_alert()  # Send email alert once
                    fire_detected = True
            else:
                label = "No Fire"
                color = (0, 255, 0)  # Green color for no fire

            # Display the label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Convert the frame to RGB (from BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            stframe.image(frame, channels="RGB")

            # Break the loop if the stop button is pressed
            if stop_button_pressed:
                break

        cap.release()
        cv2.destroyAllWindows()

