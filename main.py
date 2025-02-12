#!/usr/bin/python
# -*- coding: utf-8 -*- 

# Импортируем необходимые модули
import cv2, os
import robin.util as robin

# код человека
# 01 - Аня
# 02 - Маша
personCode = '011'
# номер теста (0 - человека нет   1 - есть в обучающем наборе 2 - нет в обучающем наборе)
testNum = '2 new'
use_algoritm = 'LBPH'
POROG = 75

# корень для тестов по человеку
personRoot = 'test_' + personCode
path = personRoot + '/training'

# проводим обучение
trainImages = robin.training(personCode, path)

# Путь к исходным фотографиям 
path = personRoot + '/' + testNum
#path = personRoot + '/training'
#!!!!!!!!!!!!
# Cписок всех фотографий по заданному пути
listphoto = os.listdir(path)
imagePaths= []
for file in listphoto:
    paste = path + '/' + file
    imagePaths.append(paste)
    
# по всем фото в папке
good = 0
bad = 0
badNoFace = 0
sumConfGood = 0
sumConfBad = 0 
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    if image is None :
        continue
    if testNum == 'training':
        faces = []
        faces.append([0,0,image.shape[1],image.shape[0]])
    else:    
        faces = robin.detectFace(image)
            
    minDistance = None
    imageWithAllFaces = image.copy()
    startImage = image
    
    minFace = None
    minConf = None
    for (x1, y1, x2, y2) in faces:
        # Если лица найдены, пытаемся распознать их
        # Функция  robin.predict в случае успешного распознавания возвращает номер и параметр confidence,
        # этот параметр указывает на уверенность алгоритма, что это именно тот человек; чем меньше параметр, тем больше уверенность
        
        cv2.rectangle(imageWithAllFaces, (x1, y1), (x2, y2), (0, 255, 0), 10)

        rImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceImage = rImage[y1: y2, x1: x2]
        if (faceImage.size == 0):
            continue
        
        if use_algoritm == 'LBPH':
            personcode, conf = robin.predict(faceImage)
        else:
            conf = robin.calcDistance(faceImage, trainImages)
          
        if conf > POROG:
            continue 
        
        if minConf == None:
            minFace = faceImage
            minConf = conf
        else:
            if minConf > conf:
                minConf = conf
                minFace = faceImage
                
    # показываем все лица
    if testNum != '0' and testNum != 'training':
        cv2.imshow("", robin.resizeImage(imageWithAllFaces,500))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    # и одно ближайшее
    if (not minFace is None):
        if testNum == 'training':
            if minConf != 0:
                print (imagePath, '\tminConf=', minConf)
        else:
            if testNum != '0':
                cv2.imshow(str(minConf), robin.resizeImage(minFace,500))
                ret = cv2.waitKey(5000)
            else:
                ret = -1    
            if ret == 32:
                sumConfGood += minConf
                good += 1
            else: 
                sumConfBad += minConf 
                bad += 1   
            cv2.destroyAllWindows()
    else:
        badNoFace += 1
if good != 0:
    print ('среднее по good ', sumConfGood/good)        
if bad != 0:
    print ('среднее по bad ', sumConfBad/bad)       
print ('good = ', good)
print ('bad = ', bad)
print ('badNoFace = ', badNoFace)
print ('процент хороших = ', good/(good+badNoFace+bad)*100)





