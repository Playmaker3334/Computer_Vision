import cv2 as cv
import numpy as np

# Cargar el modelo previamente entrenado
cvNet = cv.dnn.readNetFromTensorflow('frozen_graph.pb')

# Iniciar la captura de video
cap = cv.VideoCapture(0)  # 0 es normalmente el índice de la cámara predeterminada

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir el frame. Salir...")
        break

    # Preparar la imagen para el modelo
    blob = cv.dnn.blobFromImage(frame, size=(224, 224), swapRB=True, crop=False)
    cvNet.setInput(blob)

    # Realizar la detección
    cvOut = cvNet.forward()

    # Interpretar la salida
    probability = cvOut[0][0]
    label = "Tumor" if probability > 0.5 else "Healthy"
    confidence = probability if label == "Tumor" else 1 - probability

    # Mostrar la imagen con la predicción
    cv.putText(frame, f"{label}: {confidence * 100:.2f}%", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Dibujar un rectángulo alrededor de toda la imagen
    color = (0, 255, 0) if label == "Healthy" else (0, 0, 255)
    cv.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 2)

    # Mostrar la imagen resultante
    cv.imshow('Camera Output', frame)

    if cv.waitKey(1) == ord('q'):
        break

# Cuando todo está hecho, liberar la captura
cap.release()
cv.destroyAllWindows()




