import cv2
import numpy as np

# Definições do padrão do tabuleiro de xadrez
chessboard_size = (9, 6)
square_size = 0.025  # Tamanho de cada quadrado em metros (ajuste conforme necessário)

# Preparação dos pontos 3D do padrão do tabuleiro de xadrez
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays para armazenar pontos 3D e 2D
objpoints = []
imgpoints = []

# Captura de imagens e detecção do padrão
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
    cv2.imshow('Chessboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibração da câmera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Distância focal
focal_length = mtx[0, 0]
print("Distância focal:", focal_length)
