import cv2
import numpy as np
import pickle

# Carrega os parâmetros de calibração da câmera a partir do arquivo salvo
with open("calibration_data.pkl", "rb") as f:
    calibration_data = pickle.load(f)

mtx_left = calibration_data["mtx_left"]
dist_left = calibration_data["dist_left"]
mtx_right = calibration_data["mtx_right"]
dist_right = calibration_data["dist_right"]
R = calibration_data["R"]
T = calibration_data["T"]
Q = calibration_data["Q"]
P1 = calibration_data["P1"]
P2 = calibration_data["P2"]

# Parâmetros da câmera estereo
baseline = np.linalg.norm(T)  # distância entre câmeras em metros
focal_length = P1[0, 0]  # comprimento focal em pixels (valor calibrado)
min_object_size = 1000  # Tamanho mínimo do objeto em pixels
max_object_size = (
    5000  # Tamanho máximo do objeto em pixels (ajuste conforme necessário)
)

# Parâmetros de disparidade
block_size = 15  # Tamanho da janela do bloco. Deve ser ímpar.
min_disp = 0  # Valor mínimo possível de disparidade
num_disp = 16 * 16  # Disparidade máxima menos disparidade mínima

# Inicializa as webcams
cap1 = cv2.VideoCapture(0)  # Câmera esquerda
cap2 = cv2.VideoCapture(2)  # Câmera direita

# Cria objeto StereoBM com parâmetros customizados
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

# Retificação das câmeras
map1_left, map2_left = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R, P1, (640, 480), cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R, P2, (640, 480), cv2.CV_16SC2
)

while True:
    # Captura os quadros de ambas as câmeras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Erro: Não foi possível capturar imagens de uma ou ambas as câmeras.")
        break

    # Aplica retificação nas imagens capturadas
    frame1_rectified = cv2.remap(frame1, map1_left, map2_left, cv2.INTER_LINEAR)
    frame2_rectified = cv2.remap(frame2, map1_right, map2_right, cv2.INTER_LINEAR)

    # Converte para escala de cinza
    gray1 = cv2.cvtColor(frame1_rectified, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_rectified, cv2.COLOR_BGR2GRAY)

    # Computa o mapa de disparidade
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Normaliza disparidade para visualização
    disparity_display = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disparity_display = np.uint8(disparity_display)

    # Converte disparidade em imagem binária
    _, binary_disparity = cv2.threshold(disparity_display, 1, 255, cv2.THRESH_BINARY)

    # Encontra contornos
    contours, _ = cv2.findContours(
        binary_disparity, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Filtra contornos com base na área e encontra o maior
        valid_contours = [
            contour
            for contour in contours
            if min_object_size < cv2.contourArea(contour) < max_object_size
        ]

        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)

            # Obtém o retângulo delimitador do maior contorno
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Extrai valores de disparidade dentro do retângulo delimitador
            box_disparity = disparity[y : y + h, x : x + w]
            valid_disparity_values = box_disparity[box_disparity > 0]

            if valid_disparity_values.size > 0:
                # Calcula a disparidade média dentro do retângulo delimitador
                avg_disparity = np.mean(valid_disparity_values)

                # Calcula a distância em metros
                distance_meters = (
                    (focal_length * baseline) / avg_disparity
                    if avg_disparity > 0
                    else float("inf")
                )

                # Converte a distância para centímetros
                distance_cm = distance_meters * 100

                # Desenha um retângulo ao redor do maior objeto
                cv2.rectangle(frame1_rectified, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame2_rectified, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Exibe a distância no quadro
                cv2.putText(
                    frame1_rectified,
                    f"Distance: {distance_cm:.2f} cm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame2_rectified,
                    f"Distance: {distance_cm:.2f} cm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # Exibe os resultados
    cv2.imshow("Disparity", disparity_display)
    cv2.imshow("Left Frame with Rectangle", frame1_rectified)
    cv2.imshow("Right Frame with Rectangle", frame2_rectified)

    # Espera 33 ms para alcançar ~30 FPS
    if cv2.waitKey(33) & 0xFF == 27:
        break

# Libera os recursos
cap1.release()
cap2.release()
cv2.destroyAllWindows()
