#se importan las librerias necesarias
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import skimage.io
from skimage import io, filters
from skimage.io import imread, imshow


#se define la funcion que calcula el punto medio
def mitad(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#se define la funcion que analiza las imagenes con dos argumentos de entrada: 
#la imagen y la distancia de trabajo
def Measure(image, distance):
  #se define la variable de la distancia de trabajo como el argumento de entrada
  #se definen las constantes de distancia focal y tamano del sensor de acuerdo 
  #con los valores obtenidos anteriormente
  distcm = distance
  distfocal = 4.59
  sensorsize = 6.4
  #se multiplica la distancia x10 para convertirla a mm
  distmm=distcm*10
  #se calcula el ancho del campo de vision
  campovision=(sensorsize*distmm)/distfocal
  #la imagen se convierte a escala de grises y se difumina
  gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gris = cv2.GaussianBlur(gris, (7, 7), 0)
  #la imagen se binariza para encontrar los contornos mas facilmente
  ret, binim = cv2.threshold(gris,100,255,cv2.THRESH_BINARY)
  #se utiliza la funcion Canny para detectar los bordes
  bordes = cv2.Canny(binim, 50, 255)
  bordes = cv2.dilate(bordes, None, iterations=1)
  bordes = cv2.erode(bordes, None, iterations=1)

  #se obtienen los contornos de la imagen
  contornos = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contornos = imutils.grab_contours(contornos)
  #si se detectan multiples contornos, se omite el mas pequeno
  for c in contornos:
    if cv2.contourArea(c) < 100:
	    continue
    #se copia la imagen original para dibujar sobre ella
    orig = image.copy()
    #con los contornos se calcula una caja circunscrita a la figura
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
  cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
  
  #se calcula el punto medio de las lineas verticales para generar la linea del ancho
  (tl, tr, br, bl) = box
  (tltrX, tltrY) = mitad(tl, tr)
  (blbrX, blbrY) = mitad(bl, br)
  (tlblX, tlblY) = mitad(tl, bl)
  (trbrX, trbrY) = mitad(tr, br)
  #se dibuja la linea que da el ancho del objeto
  cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

  #se calcula el ancho en pixeles del objeto en la imagen
  anchoim = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
  #se obtiene el ancho en pixeles total de la imagen
  anchopix= image.shape[1]
  #por medio de una regla de 3, se calcula el ancho en mm del objeto, 
  #relacionando el valor del campo de vision, el ancho del objeto en pixeles 
  #y el ancho total en pixeles
  measure=(campovision*anchoim)/anchopix
  #se escribe en la imagen el valor del ancho en mm del objeto
  cv2.putText(orig, "{:.1f}mm".format(measure),(int(tltrX), int(tltrY/2)), cv2.FONT_HERSHEY_PLAIN,4, (255, 255, 255), 2, cv2.LINE_AA)

  #se define un array que contenga la imagen con los graficos y el valor de la medicion
  resultados = [orig,measure]
  return resultados
  
  #se importan las 3 imagenes de muestra
image1 = cv2.imread("/content/caja.jpeg")
image2 = cv2.imread("/content/ds.jpeg")
image3 = cv2.imread("/content/estuche.jpeg")

imagenes = np.concatenate((cv2.rotate(image1, cv2.cv2.ROTATE_90_CLOCKWISE),image2,image3),axis=1)
imshow(imagenes)

imshow(Measure(image1,16)[0])
print(Measure(image1,16)[1])

imshow(Measure(image2,16)[0])
print(Measure(image2,16)[1])

imshow(Measure(image3,16)[0])
print(Measure(image3,16)[1])
