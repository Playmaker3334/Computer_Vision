import cv2
from ultralytics import YOLO
import numpy as np
import sys

def init_model():
    """Inicializa y retorna el modelo YOLOv8 pre-entrenado."""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)

def get_available_cameras():
    """Busca cámaras disponibles."""
    available_cameras = []
    for i in range(10):  # Prueba los primeros 10 índices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Usar CAP_DSHOW en Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Cámara {i} disponible")
            cap.release()
    return available_cameras

def process_frame(frame, model):
    """Procesa un frame usando YOLOv8 y dibuja las detecciones."""
    try:
        results = model(frame)
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                if int(box.cls) == 0:  # Persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 
                              f'Persona {conf:.2f}', 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9,
                              (0, 255, 0),
                              2)
        
        return frame
    except Exception as e:
        print(f"Error al procesar el frame: {e}")
        return frame

def main():
    # Inicializamos el modelo
    model = init_model()
    
    # Buscamos cámaras disponibles
    available_cameras = get_available_cameras()
    
    if not available_cameras:
        print("No se encontraron cámaras disponibles")
        return
    
    # Usamos la primera cámara disponible
    camera_index = available_cameras[0]
    print(f"Usando cámara {camera_index}")
    
    # Inicializamos la cámara con CAP_DSHOW
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {camera_index}")
        return
    
    # Intentamos establecer una resolución común
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Presiona 'q' para salir")
    print("Presiona 'c' para cambiar de cámara")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame")
            break
            
        processed_frame = process_frame(frame, model)
        cv2.imshow('Detección de Personas', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and len(available_cameras) > 1:
            # Cambiar a la siguiente cámara disponible
            current_index = available_cameras.index(camera_index)
            next_index = (current_index + 1) % len(available_cameras)
            camera_index = available_cameras[next_index]
            
            cap.release()
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"No se pudo abrir la cámara {camera_index}")
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()