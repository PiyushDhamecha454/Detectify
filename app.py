import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Feature Detection App", layout="centered")

st.title("Detectify 🔍 - Feature Detection using OpenCV")

# Sidebar options
algo = st.selectbox("Choose Algorithm", ["SIFT", "Harris Corner", "Shi-Tomasi"])

# Helper to upload image
def upload_image(label):
    uploaded_file = st.file_uploader(label, type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        return image
    return None

# SIFT Function
def sift_match(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    return result

# Harris Corner Function
def harris_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img

# Shi-Tomasi Function
def shi_tomasi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.1, 10)
    corners = np.intp(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    return img

# UI Logic
if algo == "SIFT":
    st.subheader("SIFT Keypoint Matching")
    img1 = upload_image("Upload First Image")
    img2 = upload_image("Upload Second Image")
    if img1 is not None and img2 is not None:
        result = sift_match(img1, img2)
        st.image(result, channels="RGB", caption="SIFT Matches")

elif algo == "Harris Corner":
    st.subheader("Harris Corner Detection")
    img = upload_image("Upload Image")
    if img is not None:
        result = harris_corners(img.copy())
        st.image(result, channels="RGB", caption="Harris Corners")

elif algo == "Shi-Tomasi":
    st.subheader("Shi-Tomasi Corner Detection")
    img = upload_image("Upload Image")
    if img is not None:
        result = shi_tomasi(img.copy())
        st.image(result, channels="RGB", caption="Shi-Tomasi Corners")