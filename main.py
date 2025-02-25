import time

from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite)
from face_recognition import (face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path, makedirs)
from multiprocessing import Pool
from numpy import argmin
from string import ascii_lowercase
from random import choice
import pandas as pd
from time import strftime


class DB:
    """Classe responsável por cuidar do acesso ao banco de dados com as imagens"""
    directory = 'DB'

    def __init__(self) -> None:
        print("Iniciando o sistema... \nCarregando o Banco de Dados... ")
        self.images, self.names = [], []
        self.get_img_and_name_general()
        print("Banco carregado com sucesso... \nIniciando o encoding das imagens...")
        self.encode_list = []
        self.find_encodings()
        print("Encoding terminado com sucesso... \nSistema iniciado com sucesso")

    def get_img_and_name_general(self) -> None:
        for cl in listdir(DB.directory):
            self.images.append(imread(f'{DB.directory}/{cl}'))
            self.names.append(path.splitext(cl)[0])

    def find_encodings(self) -> None:
        with Pool(processes=None) as pool:
            self.encode_list = pool.map(self.encode_face, self.images)

    @staticmethod
    def encode_face(image) -> None:
        encoding = face_encodings(cvtColor(image, COLOR_BGR2RGB))
        if encoding:
            return encoding[0]


class FaceRecognitionSystem:
    def __init__(self, database, distance_limit):
        """
            Inicializa o sistema de reconhecimento facial.
        """
        self.dataBase = database
        self.unknown_faces_data = pd.DataFrame(columns=["ID", "Timestamp"])
        self.limite_distancia = distance_limit
        self.cap = VideoCapture(0)  # Inicializa a câmera
        self.cap.set(3, 640)  # Define a largura para 640 pixels (VGA)
        self.cap.set(4, 480)  # Define a altura para 480 pixels (VGA)
        namedWindow('Webcam')

    @staticmethod
    def generate_unique_id(length=8):
        return ''.join(choice(ascii_lowercase) for _ in range(length))

    @staticmethod
    def find_faces(img):
        """
            Encontra e codifica faces na imagem fornecida.
        """
        images = cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB)
        faces_cur_frame = face_locations(images)
        encode_cur_frame = face_encodings(images, faces_cur_frame)
        return list(zip(encode_cur_frame, faces_cur_frame))

    def process_frame(self, img):
        access_granted = False
        nome = ""

        encodings_and_locations = self.find_faces(img)

        for encodeFace, faceLoc in encodings_and_locations:
            matches = compare_faces(self.dataBase.encode_list, encodeFace, self.limite_distancia)
            distancia = face_distance(self.dataBase.encode_list, encodeFace)
            match = argmin(distancia)
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if matches[match]:
                nome = self.dataBase.names[match].upper()
                access_granted = True  # Acesso liberado se houver uma correspondência

            if distancia[match] <= self.limite_distancia:
                rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                putText(img, nome, (left, top - 10), FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Rosto desconhecido salvo!")

                time.sleep(10)

                unique_id = self.generate_unique_id()
                new_entry = pd.DataFrame({"ID": [unique_id], "Timestamp": [strftime("%Y-%m-%d %H:%M:%S")]})
                self.unknown_faces_data = pd.concat([self.unknown_faces_data, new_entry], ignore_index=True)

        return access_granted, nome

    def run(self):
        """
        Executa o sistema de reconhecimento facial em tempo real.

        Pressione 'q' para encerrar a execução.
        """
        while True:
            success, img = self.cap.read()

            access_granted, nome = self.process_frame(img)

            if access_granted:
                # Libere o acesso aqui (por exemplo, abra uma porta)
                print("Acesso Liberado")

            imshow('Webcam', img)

            if waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        destroyAllWindows()


if __name__ == '__main__':
    limite_distancia = 0.4
    myDatabase = DB()
    myFaceRecognitionSystem = FaceRecognitionSystem(myDatabase, limite_distancia)
    myFaceRecognitionSystem.run()
