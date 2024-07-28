from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

model = load_model('gender_detection.keras')

webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']

while webcam.isOpened():
    status, frame = webcam.read()

    if not status:
        break

    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):
        (startx, starty) = f[0], f[1]
        (endx, endy) = f[2], f[3]

        cv2.rectangle(frame, (startx, starty), (endx, endy), (0, 255, 0), 2)

        face_crop = np.copy(frame[starty:endy, startx:endx])
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]

        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        y = starty - 10 if starty - 10 > 10 else starty + 10

        cv2.putText(frame, label, (startx, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 2)

    cv2.imshow("gender detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
