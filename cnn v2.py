import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session() #Linpiamos todo

data_entrenamiento =  "C:\machis+\coding\prototipo_pythpn\movements prueba"
data_validacion =  "C:\machis+\coding\prototipo_pythpn\movements verificacion"

# Parametros
interacciones = 20
altura, longitud = 500, 500
batch_size = 1
pasos = 150/1
pasos_validacion = 150/1
filtrosconv1 = 32
filtrosconv2 = 64
tam_filtro1 = (4, 4)
tam_filtro2 = (3, 3)
tam_pool = (2, 2)
clases = 12
lr = 0.0005

#pre-procesamiento de las imagenes
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255, #pasar los pixeles de 0 a 255 ¦ 0 a 1
    shear_range = 0.3, #generar nuestras imagenes incluidas para un mejor entrenamiento
    horizontal_flip = True #invierte las imagenes para mejor entrenamiento
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    data_entrenamiento, #va a tomar las fotos almacenadas
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = "categorical", #clasificacion categorica por clases
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = "categorical",
)

#creamos la red neuronal convolucional
cnn = Sequential()
#agregamos filtros con el fin de volver la imagen muy profunda pero pequena
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding= 'same', input_shape = (altura, longitud, 3), activation = 'relu'))
#es una colvolucion y realizamos config
cnn.add(MaxPooling2D(pool_size = tam_pool))

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'same', activation = 'relu')) #agregamos nueva capa
cnn.add(MaxPooling2D(pool_size = tam_pool))

#vamos a convertir la imagen profunda a plana para tener la info en 1 dimension
cnn.add(Flatten()) #aplanamos la imagen
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax'))

#agregamos parametros para el modelo
#durante el entrenamiento tenga una autoevaluacion, que se optimice con Adam y la metrica sera accuracy
optimizar = tensorflow.keras.optimizers.Adam(learning_rate = lr)
cnn.compile(loss= 'categorical_crossentropy', optimizer = optimizar, metrics= ['accuracy'])

#entrenamos a nuestra red
cnn.fit(imagen_entreno, steps_per_epoch= pasos, epochs= interacciones, validation_data= imagen_validacion, validation_steps= pasos_validacion)

#guardamos el modelo
cnn.save('modelo.h5')
cnn.save_weights('pesos.h5')