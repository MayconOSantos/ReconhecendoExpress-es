import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# Dimensões das imagens de entrada
img_width, img_height = 150, 150

# Diretórios das imagens de treinamento e validação
train_data_dir = 'C:/Users/maycon.o.santos/Downloads/expressoes'
validation_data_dir = 'C:/Users/maycon.o.santos/Downloads/expressao_treino/validation'

# Número de amostras de treinamento e validação
nb_train_samples = 2478
nb_validation_samples = 2478  # Ajuste conforme necessário

# Número de épocas e tamanho do lote
epochs = 10
batch_size = 16

# Lista de classes
classes = ['surprise', 'sadness', 'neutral', 'happy', 'fear', 'disgust', 'anger']

# Configurar o gerador de dados de treinamento com aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Configurar o gerador de dados de validação (apenas rescale, sem aumento)
test_datagen = ImageDataGenerator(rescale=1./255)

# Configurar o gerador de dados de treinamento
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes)

# Configurar o gerador de dados de validação
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes)

# Verificar se os geradores encontraram imagens
print(f"Treinamento: {train_generator.samples} imagens em {train_generator.num_classes} classes")
print(f"Validação: {validation_generator.samples} imagens em {validation_generator.num_classes} classes")

# Criar o modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes)))
model.add(Activation('softmax'))

# Compilar o modelo
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# Diretório para salvar o modelo
save_dir = 'C:/Users/maycon.o.santos/Downloads/expressao_treino/modelos_salvos'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Caminho completo para salvar o modelo
model_path = os.path.join(save_dir, 'modelo_expressoes.keras')

# Salvar o modelo treinado
model.save(model_path)
print(f"Modelo salvo em: {model_path}")
