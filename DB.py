import os

from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite)
from face_recognition import (face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path, makedirs)
from multiprocessing import Pool
from numpy import argmin
from string import ascii_lowercase
from random import choice


class DB:
    """Classe responsável por cuidar do acesso ao banco de dados com as imagens"""
    directory = 'DB'
    cache_directory = 'CACHE'

    def __init__(self) -> None:
        print("Iniciando o sistema... \nCarregando o Banco de Dados... ")
        self.images, self.names, self.unknown_images = [], [], []
        self.get_img_and_name_general()
        print("Banco carregado com sucesso... \nIniciando o encoding das imagens...")
        self.encode_list, self.encode_unknown = [], []

        self.find_encodings()
        self.find_unknown_encodings()

        print("Encoding terminado com sucesso... \nSistema iniciado com sucesso")

    def get_img_and_name_general(self) -> None:

        for cl in listdir(DB.directory):
            self.images.append(imread(f'{DB.directory}/{cl}'))
            self.names.append(path.splitext(cl)[0])

    def get_img_unknowns(self):

        for x in listdir(DB.cache_directory):
            self.unknown_images.append(imread(f'{DB.cache_directory}/{x}'))


    @staticmethod
    def grant_access(name, known_face) -> None:

        makedirs('ACCESS', exist_ok=True)

        imwrite(f"ACCESS/{name}.jpg", known_face)

    @staticmethod
    def unknown_grant_access(name, known_face) -> None:

        makedirs('RD', exist_ok=True)

        imwrite(f"RD/{name}.jpg", known_face)

    @staticmethod
    def cache_access(known_face) -> None:

        makedirs('CACHE', exist_ok=True)

        imwrite("CACHE/cache.jpg", known_face)

    def find_encodings(self) -> None:
        with Pool(processes=None) as pool:
            self.encode_list = pool.map(self.encode_face, self.images)

    def find_unknown_encodings(self) -> None:
        with Pool(processes=None) as pool:
            self.encode_list = pool.map(self.encode_face, self.unknown_images)

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

                known_face = img[top:bottom, left:right] # Passando rostos conhecidos

                self.dataBase.grant_access(nome, known_face) # Salvando na pasta de rostos conhecidos que entraram

            if distancia[match] <= self.limite_distancia:
                rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                putText(img, nome, (left, top - 10), FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:

                print("Rosto desconhecido encontrado!")

                matches_unknown = compare_faces(self.dataBase.encode_unknown, encodeFace, self.limite_distancia)
                distance_unknown = face_distance(self.dataBase.encode_unknown, encodeFace)
                match_unknown = argmin(distance_unknown)

                if not matches_unknown[match_unknown]:

                    unique_id = self.generate_unique_id()
                    unknown_face = img[top:bottom, left:right]

                    self.dataBase.unknown_grant_access(unique_id, unknown_face)
                    self.dataBase.cache_access(unknown_face)

                rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)


                # time.sleep(10)

        return access_granted, nome

    def run(self):

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
