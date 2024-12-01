import cv2


class Visualizer:
    @staticmethod
    def display_results(result_queue):
        """
        Muestra los resultados procesados en tiempo real.
        """
        while True:
            if not result_queue.empty():
                frame, status = result_queue.get()
                cv2.putText(frame, f"Estado: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Detección de Hojas y Salud", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
