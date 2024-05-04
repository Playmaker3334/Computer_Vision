import cv2

# Pre-load the cascade classifier to avoid loading it in each frame which is inefficient
FaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define constants for the frame size and the minimum area for detection
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_AREA = 500
COLOR = (255, 0, 255)

# Start the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
# Control brightness if required
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)



while True:
    success, img = cap.read()
    if not success:
        break
    
    # Convert the frame to grayscale to improve detection speed and accuracy
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use GPU for the detection if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_imgGray = cv2.cuda_GpuMat()
        gpu_imgGray.upload(imgGray)
        faces = FaceCascade.detectMultiScale(gpu_imgGray)
    else:
        # Perform face detection
        faces = FaceCascade.detectMultiScale(imgGray, 1.1, 4)

    # Draw rectangles and label the faces
    for i, (x, y, w, h) in enumerate(faces, 1):
        area = w * h
        if area > MIN_AREA:
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOR, 2)
            cv2.putText(img, f"Person {i}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()

