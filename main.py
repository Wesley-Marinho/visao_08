import cv2
import numpy as np

# Defina a distância entre as duas câmeras (baseline) e a distância focal
baseline = 10.0  # Distância entre as duas câmeras em cm
focal_length = 700.0  # Distância focal da câmera em pixels (ajuste conforme necessário)

# Abra as câmeras
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Crie o objeto de correspondência estéreo
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while True:
    # Capture os frames das duas câmeras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Verifique se os frames foram capturados corretamente
    if not ret1 or not ret2:
        print("Falha ao capturar as imagens")
        break

    # Converta os frames para escala de cinza
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute o mapa de disparidade
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Calcule a profundidade (em cm) a partir do mapa de disparidade
    with np.errstate(divide="ignore"):
        depth = (baseline * focal_length) / (
            disparity + 1e-5
        )  # Para evitar divisão por zero

    # Encontre o ponto mais próximo
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(disparity)

    # Obtenha a profundidade no ponto mais próximo
    nearest_distance = depth[min_loc[1], min_loc[0]]

    # Exiba a distância na tela
    cv2.putText(
        frame1,
        f"Distancia: {nearest_distance:.2f} cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Mostre os frames e o mapa de disparidade
    cv2.circle(frame1, min_loc, 5, (0, 0, 255), -1)
    cv2.imshow("Frame 1", frame1)
    cv2.imshow("Disparity", (disparity / 16).astype(np.uint8))

    # Saia do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libere os recursos
cap1.release()
cap2.release()
cv2.destroyAllWindows()
