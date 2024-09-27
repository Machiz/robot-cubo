import cv2
import mediapipe as mp 


#leemos la camara
cap = cv2.VideoCapture(1)

#creamos un objeto que va a almacenar la deteccion y el seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()#primer parametro, FALSE para que no haga la deteccion 24/7
#solo hara deteccion cuando haya una confianza alta
#segundo parametro: numero minimo de manos
#tercer parametro confianza minima de deteccion
#cuarto parametro: confianza minima de seguimiento

#metodo para dibujar las manos
dibujo = mp.solutions.drawing_utils


while(1):
    ret, frame = cap.read()
    color  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = [] #en esta lista vamos a almacenar las coordenadas y los puntos 
    #print(resultado.multi_hand_landmarks) #si queremos ver si existe la deteccion
    
    if resultado.multi_hand_landmarks: #si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks: #buscamos la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark):
                #print(id, lm) #como nos entregan decimales (proporcion de la imagen) debemos pasarlo a pixeles
                alto, ancho, c = frame.shape #extendemos el alto y ancho de fotogramas para multiplicarlos por la proporcion 
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) !=0:
                pto_i1 = posiciones[4] #5 dedos: 4 ¦ 0 dedos: 3 ¦ 1 dedo: 2 ¦ 2 dedos: 3 ¦ 3 dedos: 4 ¦ 4 dedos: 8
                pto_i2 = posiciones[20] #5 dedos: 20 ¦ 0 dedos: 17 ¦ 1 dedo: 17¦ 2 dedos: 20 ¦ 3 dedos: 20 ¦ 4 dedos: 20
                pto_i3 = posiciones[12] #5 dedos: 12 ¦ 0 dedos: 10 ¦ 1 dedo: 20 ¦ 2 dedos: 16 ¦ 3 dedos: 12 ¦ 4 dedos: 12
                pto_i4 = posiciones[0] #5 dedos: 0 ¦ 0 dedos: 0 ¦ 1 dedo: 0 ¦ 2 dedos: 0 ¦ 3 dedos: 0 ¦ 4 dedos: 0
                pto_i5 = posiciones[9] #punto central
                x1, y1 = (pto_i5[1] - 250), (pto_i5[2] - 250) #obtenemos el inicial y las longitudes
                ancho, alto = (x1 + 500), (y1 + 500)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            #dedos_reg = cv2.resize(dedos_reg, (500, 500), interpolation = cv2.INTER_CUBIC) #Redimensionamos las fotos
            #cv2.imwrite(carpeta + "/izquierda{}.jpg".format(cont), dedos_reg)
            #cont = cont + 1

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
cap.release()
cv2.destroyALLWindows()