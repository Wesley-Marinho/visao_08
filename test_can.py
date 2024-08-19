import cv2

cap = cv2.VideoCapture(
    0
)  # Tente alterar o índice para 1, 2, etc., se houver várias câmeras

if not cap.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar a imagem")
        break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
