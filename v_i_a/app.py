
from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np

app = Flask(__name__, static_url_path='/static')
def process_image(image_path, reference_dir):
    # Load the uploaded image
    uploaded_image = cv2.imread(image_path)
    if uploaded_image is None:
        return "Error: Uploaded image could not be loaded."

    # Convert to grayscale and apply Gaussian blur
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    uploaded_blur = cv2.GaussianBlur(uploaded_gray, (5, 5), 0)

    result = "OK"
    good_match_threshold = 10
    color_difference_threshold = 1000

    # Initialize ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(uploaded_blur, None)

    if des1 is None:
        return "NG"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for ref_file in os.listdir(reference_dir):
        ref_path = os.path.join(reference_dir, ref_file)
        reference_image = cv2.imread(ref_path)
        if reference_image is None:
            continue

        # Convert reference image to grayscale and apply Gaussian blur
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        reference_blur = cv2.GaussianBlur(reference_gray, (5, 5), 0)

        # Resize both images to the same dimensions
        height, width = uploaded_blur.shape[:2]
        reference_resized = cv2.resize(reference_blur, (width, height))

        # Compute keypoints and descriptors for the resized reference image
        kp2, des2 = orb.detectAndCompute(reference_resized, None)
        if des2 is None:
            continue

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]

        # Check for color differences using Mean Squared Error (MSE)
        mse_value = np.mean((uploaded_blur.astype("float") - reference_resized.astype("float")) ** 2)

        if len(good_matches) >= good_match_threshold and mse_value < color_difference_threshold:
            matched_img = cv2.drawMatches(uploaded_blur, kp1, reference_resized, kp2, good_matches, None, flags=2)
            debug_path = f'static/debug/matched_{ref_file}'
            cv2.imwrite(debug_path, matched_img)
            return "OK"

    return "NG"

'''
def process_image(image_path, reference_dir):
    # Load the uploaded image
    uploaded_image = cv2.imread(image_path)
    if uploaded_image is None:
        return "Error: Uploaded image could not be loaded."

    # Convert to grayscale and apply Gaussian blur
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    uploaded_blur = cv2.GaussianBlur(uploaded_gray, (5, 5), 0)

    result = "OK"
    good_match_threshold = 10  # Minimum number of good matches for "OK"
    color_difference_threshold = 1000  # Maximum allowable pixel differences (adjust as needed)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Compute keypoints and descriptors for the uploaded image
    kp1, des1 = orb.detectAndCompute(uploaded_blur, None)

    if des1 is None:
        return "NG"  # No descriptors found in the uploaded image

    # BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for ref_file in os.listdir(reference_dir):
        ref_path = os.path.join(reference_dir, ref_file)
        reference_image = cv2.imread(ref_path)
        if reference_image is None:
            continue

        # Convert reference image to grayscale and apply Gaussian blur
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        reference_blur = cv2.GaussianBlur(reference_gray, (5, 5), 0)

        # Compute keypoints and descriptors for the reference image
        kp2, des2 = orb.detectAndCompute(reference_blur, None)
        if des2 is None:
            continue  # Skip if no descriptors are found in the reference image

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Evaluate matches
        good_matches = [m for m in matches if m.distance < 50]

        # Check for color differences using Mean Squared Error (MSE)
        mse_value = np.mean((uploaded_blur.astype("float") - reference_blur.astype("float")) ** 2)
        
        if len(good_matches) >= good_match_threshold and mse_value < color_difference_threshold:
            # Debugging: Save matched keypoints image
            matched_img = cv2.drawMatches(uploaded_blur, kp1, reference_blur, kp2, good_matches, None, flags=2)
            debug_path = f'static/debug/matched_{ref_file}'
            cv2.imwrite(debug_path, matched_img)
            return "OK"

    return "NG"
'''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Handle image upload and processing
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    uploads_dir = 'static/uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Reference images directory
    reference_dir = 'static/reference_images'

    # Process the uploaded image
    result = process_image(file_path, reference_dir)

    # Choose the appropriate image for the result
    result_image = 'green_tick.png' if result == "OK" else 'red_cross.png'

    return render_template('result.html', result=result, image_path=file.filename, result_image=result_image)

@app.route('/capture', methods=['POST'])
def capture():
    # Capture image from USB camera
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return "Error: Could not access camera."

    # Save the captured image
    capture_dir = 'static/uploads'
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    capture_path = os.path.join(capture_dir, 'captured_image.jpg')
    cv2.imwrite(capture_path, frame)

    # Reference images directory
    reference_dir = 'static/reference_images'

    # Process the captured image
    result = process_image(capture_path, reference_dir)

    # Choose the appropriate image for the result
    result_image = 'green_tick.png' if result == "OK" else 'red_cross.png'

    return render_template('result.html', result=result, image_path='captured_image.jpg', result_image=result_image)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

