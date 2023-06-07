import tensorflow as tf
from django.conf import settings
from PIL import Image
from tensorflow import keras
from rest_framework.response import Response
import numpy as np

class Preditor:
    def __init__(self):
        self.model = tf.saved_model.load(str(settings.BASE_DIR / 'melhor_modelo'))
        #self.model = keras.model.load_model('melhor_modelo.h5')


    def preprocess_imagem(self, imagem):
        imagem = Image.open(imagem)
        imagem = imagem.resize((224, 224))
        imagem = imagem.convert('L')  # Converte para escala de cinza
        imagem = tf.keras.preprocessing.image.img_to_array(imagem)
        imagem = tf.expand_dims(imagem, axis=0)
        imagem = imagem / 255.0  # Normaliza os valores dos pixels entre 0 e 1
        return imagem

   # def prever(self, imagem):
        imagem = self.preprocess_imagem(imagem)
        imagem_np = imagem.numpy()
        #resultado = self.model.predict(imagem)
        resultado = self.model(imagem_np)
        # Processamento do resultado da predição
        resultado_lista = resultado.tolist()

        #return resultado
        return Response(resultado_lista)

    def prever(self, imagem):
        # ...
        imagem = self.preprocess_imagem(imagem)
        # Converter EagerTensor para uma matriz NumPy
        imagem_np = imagem.numpy()

        # Realizar a predição com o modelo
        resultado = self.model(imagem_np)

        # Aplicar a função sigmoide
        #resultado_sigmoide = tf.math.sigmoid(resultado)

        # Aplicar o limiar de 0.5
        resultado_limiar = tf.where(resultado >= 0.5, 1, 0)

        # Converter o resultado em uma lista Python nativa
        resultado_lista = np.array(resultado_limiar).tolist()

        # Extrair os valores individuais da lista
        classe1 = resultado_lista[0][0]
        classe2 = resultado_lista[0][1]
        classe3 = resultado_lista[0][2]
        classe4 = resultado_lista[0][3]
        classe5 = resultado_lista[0][4]

        retorno = {
            'classe1': classe1,
            'classe2': classe2,
            'classe3': classe3,
            'classe4': classe4,
            'classe5': classe5
        }

        # Retornar o dicionário na resposta da API
        return retorno

