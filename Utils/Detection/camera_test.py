import cv2
#Initially created to test camera on raspberry pi module
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot open camera")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Camera Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
