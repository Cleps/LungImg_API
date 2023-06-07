from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from .modelo import Preditor

class PredicaoView(APIView):
    def post(self, request, format=None):
        imagem = request.FILES.get('imagem')
        if imagem:
            preditor = Preditor()
            resultado = preditor.prever(imagem)
            return Response(resultado)
        return Response({'error': 'Nenhuma imagem enviada.'})
