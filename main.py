import cv2
import numpy as np

# Configurações para as câmeras
camera_left = cv2.VideoCapture(0)  # ID da primeira câmera
camera_right = cv2.VideoCapture(2)  # ID da segunda câmera

# Verificar se as câmeras estão abertas
if not camera_left.isOpened() or not camera_right.isOpened():
    print("Erro ao abrir as câmeras.")
    exit()

# Configurações para o cálculo do mapa de disparidade
block_size = 5  # Tamanho do bloco para correspondência
min_disp = 0  # Disparidade mínima
num_disp = 16  # Número de disparidades

# Criar objeto StereoBM
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

while True:
    # Captura dos quadros das duas câmeras
    ret_left, frame_left = camera_left.read()
    ret_right, frame_right = camera_right.read()

    if not ret_left or not ret_right:
        print("Falha ao capturar imagens das câmeras.")
        break

    # Converter as imagens para escala de cinza
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Calcular o mapa de disparidade
    disparity = stereo.compute(gray_left, gray_right)

    # Normalizar o mapa de disparidade para exibição
    disparity = cv2.normalize(
        disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disparity = np.uint8(disparity)

    # Exibir as imagens capturadas e o mapa de disparidade
    cv2.imshow("Esquerda", frame_left)
    cv2.imshow("Direita", frame_right)
    cv2.imshow("Mapa de Disparidade", disparity)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar as câmeras e fechar as janelas
camera_left.release()
camera_right.release()
cv2.destroyAllWindows()
