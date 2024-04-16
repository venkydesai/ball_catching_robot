
###Save Images
import cv2

# Create a video capture object
cap = cv2.VideoCapture(2)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define the codec and create a VideoWriter object if you want to save the video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
value = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        value += 1
        cv2.imwrite('./image/captured_'+str(value)+".jpg", frame)
        print("Image saved."+str(value))

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
# out.release() # Uncomment this line if you are saving the video
cv2.destroyAllWindows()
