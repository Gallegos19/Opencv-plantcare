import cv2
import threading
import queue


class FrameHandler:
    def __init__(self, max_queue_size=10):
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)

    def capture_frames(self, cap):
        """
        Captura fotogramas de la cámara y los coloca en la cola de captura.
        """
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el fotograma.")
                break

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_frames(self, plant_detector):
        """
        Procesa los fotogramas utilizando el detector de plantas y coloca los resultados en la cola de resultados.
        """
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame, status = plant_detector.analyze_frame(frame)
                self.result_queue.put((processed_frame, status))
