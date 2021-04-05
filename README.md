<!-- Visualizador online: https://stackedit.io/ -->
 ![Logo da UDESC Alto Vale](http://www1.udesc.br/imagens/id_submenu/2019/marca_alto_vale_horizontal_assinatura_rgb_01.jpg)
 
 ---
 
# Algoritmo de Rede Neural Convolucional para o reconhecimento e classificação de imagens

Algoritmo desenvolvido no âmbito acadêmico para a disciplina de Inteligência Artificial do [Centro de Educação Superior do Alto Vale do Itajaí (CEAVI/UDESC)](https://www.udesc.br/ceavi).

# Autor
- [**Leonardo Tadeu Jaques Steinke**](mailto:leonardosteinke1@gmail.com) - [LeonardoSteinke](https://github.com/LeonardoSteinke)

# Sumário

* [Problema](#problema)
* [Dataset](#dataset)
* [Técnica](#tecnica)
* [Resultados Obtidos](#resultados)
* [Instruções para Uso do Software](#instrucoes)
* [Vídeo](#video)

## [Problema](#problema)
O Problema a ser desenvolvido é uma Rede Neural Convolucional para o de reconhecimento e classificação binária de imagens.

## [Dataset](#dataset)
O Dataset utilizado será uma versão adaptada do [Kaggle](https://www.kaggle.com/) da competição [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) contendo 7000 imagens de Cachorros e 7000 imagens de Gatos.

## [Técnica](#tecnica)
Utilização da Rede Neural CNN para reconhecer e classificar de forma binária imagens de Gatos e Cachorros

### Bibliotecas utilizadas
- [TensorFlow](https://www.tensorflow.org/?hl=pt-br)
- [Keras](https://keras.io/) 
- [OpenCV](https://opencv.org/)
- [MatplotLib](https://matplotlib.org/)

### Entradas e Saídas
- O dataset de treino é colocado em uma pasta no mesmo diretório chamada "treino"
- As imagens de teste são colocadas em uma pasta no mesmo diretório chamada "teste"

### Critério de Parada
- O algoritmo termina quando é atingido o numero de épocas definida pelo usuário
- O número de épocas pode ser definida pela variável "Epocas" no codigo fonte

## [Resultados Obtidos](#resultados)
Para medir os resultados foram realizados os seguintes testes
- 1 Época de treinamento
- 10 Épocas de treinamento
- 25 Épocas de treinamento
- 50 Épocas de treinamento

A rede foi executada por um total de 50 épocas, o que durou um período de aproximadamente 133 minutos, sendo em média 160,52 segundos cada época do treinamento como mostra a figura abaixo![Treinamento 50 Epocas](https://user-images.githubusercontent.com/26045336/113521202-47e40880-956e-11eb-813a-bf97bbf40de4.png)


Validações foram feitas retirando imagens do proprio Dataset, imagens retiradas da Internet e Imagens Próprias

![Perda de treino e validação 10 epocas](https://user-images.githubusercontent.com/26045336/113520620-ac04cd80-956a-11eb-9c3d-54ac8fd17fae.png)

Grafico de Perda de treino e Validação para 10 épocas


![Perda de treino e validação 25 epocas](https://user-images.githubusercontent.com/26045336/113520996-0c950a00-956d-11eb-9ff7-d09c9301cc57.png)

Grafico de Perda de treino e Validação para 25 épocas


![Perda de treino e validação 50 epocas](https://user-images.githubusercontent.com/26045336/113521001-11f25480-956d-11eb-8876-859f11325e39.png)

Grafico de Perda de treino e Validação para 50 épocas

É possivel verificar que quanto maior o número de épocas treinadas, maior é a acurácia da Rede Neural
- O melhor resultado dentro dessas 50 épocas de treinamento foi na época 49 onde obteve 87,14% de acurácia

## [Instruções para Uso do Software](#instrucoes)

O software pode ser executado abrindo os arquivos .py na sua IDE de preferência, (foram testadas nas IDEs [pyCharm](https://www.jetbrains.com/pt-br/pycharm/download/) e [Spyder](https://www.spyder-ide.org/)) o usuário além de ter em mãos os códigos fonte, deve criar uma pasta chamada "treino" (para colocar o dataset de treino), e uma pasta "teste" (com todas as imagens que o usuário quer classificar, ambas as pastas no mesmo diretório.

Para definir a quantidade de Épocas a serem treinadas deve-se alterar o valor da variável "Epocas" e criar um arquivo com o formato padrão "modelo+NUMERO DE ÉPOCAS+epocas.h5", caso ele ainda não exista, onde Número de épocas é o mesmo definido dentro do codigo fonte

Para a classificação das imagens deve-se colocar todas as imagens que se que classificar dentro da pasta "teste" e no codigo fonte do arquivo "Teste.py" alterar as variaveis "ImagensParaAvaliar" para a quantidade de imagens a serem avaliadas e a variável "Epocas" para o valor do modelo de épocas que você quer utilizar

## [Vídeo](#video)

[Trabalho Implementação 75INC - Rede Neural Convolucional para classificação binária de imagens](https://www.youtube.com/watch?v=lgv-l8YUhV0)



