import cv2
import numpy as np


class PlantDetector:
    def __init__(self):
        # Rango de colores para identificar áreas verdes (hojas saludables)
        self.green_lower = np.array([35, 50, 50], dtype=np.uint8)
        self.green_upper = np.array([85, 255, 255], dtype=np.uint8)

        # Rango de colores para identificar áreas marrones (hojas secas o tallos)
        self.brown_lower = np.array([10, 50, 50], dtype=np.uint8)
        self.brown_upper = np.array([20, 255, 200], dtype=np.uint8)

        # Rango de colores para identificar áreas amarillas (indicadores de enfermedad)
        self.yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
        self.yellow_upper = np.array([30, 255, 255], dtype=np.uint8)

    def detect_leaves(self, frame):
        """
        Detecta hojas en el fotograma utilizando contornos en áreas verdes.

        Args:
            frame (numpy.ndarray): Fotograma capturado por la cámara.

        Returns:
            numpy.ndarray: Máscara binaria con las áreas detectadas como hojas.
            list: Lista de contornos detectados.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_frame, self.green_lower, self.green_upper)

        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return green_mask, contours

    def analyze_frame(self, frame):
        """
        Analiza un fotograma y determina el estado de salud de las hojas.

        Args:
            frame (numpy.ndarray): Fotograma capturado por la cámara.

        Returns:
            numpy.ndarray: Fotograma procesado con el estado de salud.
            str: Estado general ("Saludable", "Enferma", "No es una planta").
        """
        green_mask, contours = self.detect_leaves(frame)

        if len(contours) == 0:
            return frame, "No es una planta"

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        is_sick = False

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = hsv_frame[y:y + h, x:x + w]
            yellow_mask = cv2.inRange(roi, self.yellow_lower, self.yellow_upper)
            brown_mask = cv2.inRange(roi, self.brown_lower, self.brown_upper)

            combined_mask = cv2.bitwise_or(yellow_mask, brown_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            infected_pixels = cv2.countNonZero(combined_mask)
            infected_percentage = (infected_pixels / total_pixels) * 100

            if infected_percentage > 5:
                is_sick = True
                cv2.putText(frame, "Enferma", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Saludable", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, "Enferma" if is_sick else "Saludable"
