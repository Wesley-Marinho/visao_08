import cv2
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
import os

# Define o diretório onde as imagens serão salvas
save_directory = "capturas"
os.makedirs(save_directory, exist_ok=True)


def update_frame():
    # Captura a imagem da webcam
    ret, frame = cap.read()
    if ret:
        # Converte a imagem do OpenCV (BGR) para o formato PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Atualiza a imagem na interface
        lbl_video.imgtk = imgtk
        lbl_video.config(image=imgtk)

    # Atualiza o frame a cada 10ms
    lbl_video.after(10, update_frame)


def capture_image():
    # Captura a imagem da webcam
    ret, frame = cap.read()
    if ret:
        # Gera um nome automático baseado na data e hora
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"imagem_{timestamp}.jpg"
        file_path = os.path.join(save_directory, file_name)

        # Salva a imagem em um arquivo
        cv2.imwrite(file_path, frame)
        print(f"Imagem salva em {file_path}")


# Configura a captura da webcam
cap = cv2.VideoCapture(0)

# Cria a janela principal
root = tk.Tk()
root.title("Captura de Imagem")

# Cria um widget de imagem para exibir a webcam
lbl_video = tk.Label(root)
lbl_video.pack()

# Cria um botão para capturar a imagem
btn_capture = tk.Button(root, text="Capturar Imagem", command=capture_image)
btn_capture.pack()

# Inicia a atualização dos frames da webcam
update_frame()

# Inicia a interface gráfica
root.mainloop()

# Libera a câmera quando a janela é fechada
cap.release()
cv2.destroyAllWindows()
