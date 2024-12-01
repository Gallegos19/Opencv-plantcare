import cv2
import threading
from plant_detector import PlantDetector
from frame_handler import FrameHandler
from visualizer import Visualizer

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    frame_handler = FrameHandler()
    plant_detector = PlantDetector()

    # Crear hilos para captura, análisis y visualización
    capture_thread = threading.Thread(target=frame_handler.capture_frames, args=(cap,), daemon=True)
    process_thread = threading.Thread(target=frame_handler.process_frames, args=(plant_detector,), daemon=True)
    display_thread = threading.Thread(target=Visualizer.display_results, args=(frame_handler.result_queue,), daemon=True)

    # Iniciar hilos
    capture_thread.start()
    process_thread.start()
    display_thread.start()

    # Esperar a que termine el hilo de visualización
    display_thread.join()
    cap.release()


if __name__ == "__main__":
    main()
