import cv2
import numpy as np
import os

# Для детектирования лиц используем каскады Хаара
faceCascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")

# или нейросеть
# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"
# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# Для распознавания лиц используем локальные бинарные шаблоны
recognizer = cv2.face.LBPHFaceRecognizer_create()

def detectHaarFace(image):
    if (image is None or len(image) == 0) :
        return []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = []
    hfaces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in hfaces:
        faces.append([x, y, x+w, y+h])
    return faces

# функция определения лиц
def detectNetFace(net, image, conf_threshold):
    height, width, channel = image.shape
    # преобразуем картинку в двоичный пиксельный объект
    blob=cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц    image.

    detections=net.forward()
    # переменная для рамок вокруг лица
    faceBoxes=[]
    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence=detections[0,0,i,2]
        # если результат превышает порог срабатывания — это лицо
        if confidence>conf_threshold:
            # формируем координаты рамки
            x1=int(detections[0,0,i,3]*width)
            y1=int(detections[0,0,i,4]*height)
            x2=int(detections[0,0,i,5]*width)
            y2=int(detections[0,0,i,6]*height)
            # добавляем их в общую переменную
            faceBoxes.append((x1,y1,x2,y2))

    # возвращаем рамки с лицами
    return faceBoxes

def detectFace(image, threshold = 0.997):
    # лица смотрят  прямо в камеру
    f1 = detectHaarFace(image)
    # сбоку
    #f2 = detectNetFace(faceNet,image, threshold)
    # объединяем вместе
    faces = f1  # + f2
    return faces

def resizeImage(image, xSize):
    desired_width = xSize  # желаемая ширина
    # соотношение сторон: ширина, делённая на ширину оригинала
    aspect_ratio = desired_width / image.shape[1]
    # желаемая высота: высота, умноженная на соотношение сторон
    desired_height = int(image.shape[0] * aspect_ratio)
    dim = (desired_width, desired_height)  # итоговые размеры
    # Масштабируем картинку
    resizedImage = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_AREA)
    return resizedImage
    #image.

def renameImages(fullPath, code):
    imagePathsAll = os.listdir(fullPath)
    i = 1
    for f in imagePathsAll:
        oldName = os.path.join(fullPath,f)
        newName =  os.path.join(fullPath,f'{code}_{i}.jpg')
        os.rename(oldName, newName )
        i = i + 1

def getImagesForLBP(personCode,path):
    # Ищем все фотографии и записываем их в image_paths
    imagePaths = []
    
    images = []
    labels = []
    
    imagePathsAll = os.listdir(path)
    for f in imagePathsAll:
        imagePaths.append(os.path.join(path, f))

    for imagePath in imagePaths:
        # читаем изображение
        image = cv2.imread(imagePath)
        if image is None :
            continue
        person = int(personCode)
        
        # Определяем области где есть лица
        #faces = detectFace(image.ge, 0.6)
        faces = []
        faces.append((0, 0, image.shape[1], image.shape[0]))

        # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels
        for (x1, y1, x2, y2) in faces:
            faceImage = image[y1: y2, x1: x2]
            if (faceImage.size == 0) :
                continue

            print (f'path = {imagePath}')        

            #cv2.imshow("", resizeImage(faceImage, 300))
            #cv2.waitKey(500)
            #cv2.destroyAllWindows()            

            # Для распознавания только оттенки серого
            faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
            
            #cv2.imshow("", faceImage)
            #cv2.waitKey(1000)
            #cv2.destroyAllWindows() 
            
            images.append(faceImage)
            labels.append(person)
    return images, labels


def orb(img) :
    # create SIFT feature extractor
    #sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    # detect features from the image
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def training(personCode, path):
    # Получаем лица и соответствующие им номера
    images, labels = getImagesForLBP(personCode, path)
    # Обучаем программу распознавать лица
    recognizer.train(images, np.array(labels))
    # Save the trained model to a file
    #recognizer.write(trainModelPath)
    #recognizer.read(trainModelPah)

    return images

def calcDistance(image, trainImages) :
    keypoints, descriptors = orb(image)
    minDistance = 100000
    
    for trainImage in trainImages :
        trainKeypoints, trainDescriptors = orb(trainImage)
        if len(trainKeypoints) == 0 or len(trainDescriptors) == 0 :
            continue
        # create feature matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
        #bf = cv2.BFMatcher()
        # match descriptors of both images
        matches = bf.match(descriptors,trainDescriptors)
        #matches = bf.knnMatch(descriptors,descriptorsOrig,k=2)
        distance = 10000
        for m in matches:
            # на одной фото много похожих мест - берем одно минимальное
            if distance > m.distance :
                distance = m.distance
        # по всем фото берем минимальное расстояние
        if minDistance > distance :
            minDistance = distance
    return minDistance    

def predict(faceImage):
    return recognizer.predict(faceImage)