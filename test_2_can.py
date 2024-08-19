import cv2

# Acesse as câmeras
cap1 = cv2.VideoCapture(0)  # Primeira câmera
cap2 = cv2.VideoCapture(2)  # Segunda câmera

# Verifique se as câmeras foram abertas corretamente
if not cap1.isOpened() or not cap2.isOpened():
    print("Não foi possível acessar as câmeras")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Erro na captura dos frames")
        break

    # Exiba os frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    # Saia com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
