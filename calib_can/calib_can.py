import cv2
import numpy as np
import pickle
import glob

# Configurações da calibração
chessboard_size = (7, 7)  # Número de cruzes no padrão de xadrez (padrão 8x8)
square_size = 0.025  # Tamanho do quadrado no padrão de xadrez em metros (ajuste conforme necessário)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara os pontos de objeto, como (0,0,0), (1,0,0), (2,0,0), ... (6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays para armazenar pontos de objeto e pontos de imagem de todas as imagens.
objpoints = []  # Pontos 3D no espaço do mundo real
imgpoints_left = []  # Pontos 2D no espaço da imagem da câmera esquerda
imgpoints_right = []  # Pontos 2D no espaço da imagem da câmera direita

# Caminhos para as imagens das câmeras esquerda e direita
images_left = glob.glob("calib_can/calib_fotos/esquerda*.jpg")
images_right = glob.glob("calib_can/calib_fotos/direita*.jpg")

# Certifique-se de que há o mesmo número de imagens para ambas as câmeras
if len(images_left) != len(images_right):
    print("Erro: Diferente número de imagens nas pastas esquerda e direita.")
    exit()

# Captura de imagens do padrão de xadrez
for img_left, img_right in zip(images_left, images_right):
    frame1 = cv2.imread(img_left)
    frame2 = cv2.imread(img_right)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Procura pelo padrão de xadrez
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    # Se encontrado, adiciona os pontos de objeto e imagem
    if ret1 and ret2:
        objpoints.append(objp)

        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners1)

        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners2)

        # Desenha e exibe os cantos
        cv2.drawChessboardCorners(frame1, chessboard_size, corners1, ret1)
        cv2.drawChessboardCorners(frame2, chessboard_size, corners2, ret2)
        cv2.imshow("Frame Esquerdo", frame1)
        cv2.imshow("Frame Direito", frame2)

        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibração das câmeras esquerda e direita
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray1.shape[::-1], None, None
)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray2.shape[::-1], None, None
)

# Estereo calibração
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
    gray1.shape[::-1],
    criteria_stereo,
    flags,
)

# Cálculo das retificações e matriz Q
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, gray1.shape[::-1], R, T, alpha=1
)

# Salva os parâmetros em um arquivo
calibration_data = {
    "mtx_left": mtx_left,
    "dist_left": dist_left,
    "mtx_right": mtx_right,
    "dist_right": dist_right,
    "R": R,
    "T": T,
    "Q": Q,
    "P1": P1,
    "P2": P2,
}

with open("calib_can/calib_files/calibration_data.pkl", "wb") as f:
    pickle.dump(calibration_data, f)

print("Calibração concluída e salva em 'calib_can/calib_files/calibration_data.pkl'")
