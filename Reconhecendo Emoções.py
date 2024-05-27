import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
model_path = 'C:/Users/maycon.o.santos/Downloads/expressao_treino/modelos_salvos/modelo_expressoes.keras'
model = load_model(model_path)

# Definir as classes de expressão
#classes = ['surprise', 'sadness', 'neutral', 'happy', 'fear', 'disgust', 'anger']
classes = ['surpresa', 'tristeza', 'neutro', 'feliz', 'medo', 'desgosto', 'raiva']


# Inicializar o objeto de captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Capturar um quadro de vídeo
    ret, frame = cap.read()

    if not ret:
        break

    # Detectar rostos no quadro
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada rosto detectado, fazer a previsão da emoção
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (150, 150))  # Ajuste o tamanho de acordo com a forma de entrada esperada

        # Adicionar canais de cor ausentes
        resized_with_channels = np.expand_dims(resized, axis=0)

        normalized = resized_with_channels / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))  # Ajuste a forma de acordo com a entrada esperada

        result = model.predict(reshaped)

        # Obter a classe prevista
        label = np.argmax(result)
        emotion = classes[label]

        # Desenhar um retângulo em volta do rosto e mostrar a emoção prevista
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar o quadro final com a emoção prevista
    cv2.imshow('Emotion Detection', frame)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o objeto de captura de vídeo e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
