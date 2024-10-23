import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image

# Function to encode face
def encode_face(image):
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        return face_recognition.face_encodings(image, face_locations)[0]
    return None

# Onboarding function to register a new user
def onboard_user(registered_faces, name, image):
    encoding = encode_face(image)
    if encoding is not None:
        registered_faces[name] = encoding
        return True
    return False

# Verification function to compare new image to stored encodings
def verify_user(registered_faces, image):
    encoding = encode_face(image)
    if encoding is None:
        return None
    for name, stored_encoding in registered_faces.items():
        distance = face_recognition.face_distance([stored_encoding], encoding)[0]
        if distance < 0.6:  # Threshold for match
            return name, distance
    return None, None

# Streamlit app
st.title("Face Recognition System")

# Dictionary to store registered faces (in a real app, this would be saved in a database)
registered_faces = {}

# Onboarding section
st.subheader("Onboard a new user")
name = st.text_input("Enter your name:")
uploaded_image = st.file_uploader("Upload a face image for onboarding", type=["jpg", "png", "jpeg"])

if uploaded_image and name:
    image = np.array(Image.open(uploaded_image))
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Onboarding Image", width=250)  # Adjust the width to your desired size

    onboarded = onboard_user(registered_faces, name, image)
    if onboarded:
        st.success(f"User {name} has been successfully onboarded.")
    else:
        st.error("Could not detect a face. Please upload a clear face image.")

# Verification section
st.subheader("Verify a user")
verify_image = st.file_uploader("Upload a face image for verification", type=["jpg", "png", "jpeg"])

if verify_image:
    image = np.array(Image.open(verify_image))
    
    # Display the uploaded image for verification
    st.image(image, caption="Uploaded Verification Image", width=250)  # Adjust the width to your desired size
    
    name, distance = verify_user(registered_faces, image)
    if name:
        st.success(f"Verification successful! Hello, {name}. (Distance: {distance:.2f})")
    else:
        st.error("Verification failed. No match found.")