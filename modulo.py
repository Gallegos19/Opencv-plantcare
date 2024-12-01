import cv2
import numpy as np

class PlantHealthDetector:
    def __init__(self):
        # Rango de colores para identificar áreas enfermas (amarillo y marrón)
        self.yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
        self.yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
        self.brown_lower = np.array([10, 50, 50], dtype=np.uint8)
        self.brown_upper = np.array([20, 255, 200], dtype=np.uint8)

        # Rango de colores para identificar áreas verdes (hojas saludables)
        self.green_lower = np.array([35, 50, 50], dtype=np.uint8)
        self.green_upper = np.array([85, 255, 255], dtype=np.uint8)

    def detect_leaves(self, frame):
        """
        Detecta hojas en el fotograma utilizando contornos en áreas verdes.

        Args:
            frame (numpy.ndarray): Fotograma capturado por la cámara.

        Returns:
            numpy.ndarray: Máscara binaria con las áreas detectadas como hojas.
            list: Lista de contornos detectados.
        """
        # Convertir el fotograma a espacio HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear una máscara para detectar áreas verdes
        green_mask = cv2.inRange(hsv_frame, self.green_lower, self.green_upper)

        # Aplicar un filtro para reducir ruido
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # Detectar contornos en la máscara
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return green_mask, contours

    def analyze_frame(self, frame):
        """
        Analiza las hojas detectadas para determinar si están enfermas.

        Args:
            frame (numpy.ndarray): Fotograma capturado por la cámara.

        Returns:
            numpy.ndarray: Fotograma con resultados visuales.
            str: Estado general del análisis ("Saludable", "Enferma", "No es una planta").
        """
        # Detectar hojas en el fotograma
        green_mask, contours = self.detect_leaves(frame)

        if len(contours) == 0:
            return frame, "No es una planta"

        # Analizar cada contorno detectado
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        is_sick = False

        for contour in contours:
            # Dibujar el contorno en el fotograma original
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crear una región de interés (ROI) basada en el contorno
            roi = hsv_frame[y:y + h, x:x + w]

            # Detectar áreas amarillas y marrones en la ROI
            yellow_mask = cv2.inRange(roi, self.yellow_lower, self.yellow_upper)
            brown_mask = cv2.inRange(roi, self.brown_lower, self.brown_upper)
            combined_mask = cv2.bitwise_or(yellow_mask, brown_mask)

            # Calcular el porcentaje de área enferma dentro de la ROI
            total_pixels = roi.shape[0] * roi.shape[1]
            infected_pixels = cv2.countNonZero(combined_mask)
            infected_percentage = (infected_pixels / total_pixels) * 100

            # Si el área enferma supera el umbral, se marca como enferma
            if infected_percentage > 5:  # Umbral de 5%
                is_sick = True
                cv2.putText(frame, "Enferma", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Saludable", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Devolver el estado general
        status = "Enferma" if is_sick else "Saludable"
        return frame, status

    def start_camera(self):
        """
        Inicia la cámara y analiza en tiempo real si las hojas están saludables o enfermas.
        """
        cap = cv2.VideoCapture(0)  # Abre la cámara (cámara predeterminada)

        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara.")
            return

        print("Presiona 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el fotograma.")
                break

            # Analizar el fotograma
            analyzed_frame, status = self.analyze_frame(frame)

            # Mostrar el estado general en la ventana
            cv2.putText(analyzed_frame, f"Estado: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mostrar el fotograma analizado
            cv2.imshow("Detección de Hojas y Salud", analyzed_frame)

            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()


# Ejemplo de uso
if __name__ == "__main__":
    detector = PlantHealthDetector()
    detector.start_camera()
