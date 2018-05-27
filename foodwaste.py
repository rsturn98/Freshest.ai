import pygame
from pygame.locals import *
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
from pprint import pprint
import time

with open('foodDatabase.json') as f:
    database = json.load(f)

def getCategory(classId):
    if str(classId) in database:
        return database[str(classId)]['name']
    return ''

def getImagePath(classId, index):
    if str(classId) in database:
        return database[str(classId)]['image' + str(index)]
    return ''

camera = cv2.VideoCapture(0)

screenWidth = 800
screenHeight = 600

pygame.init()
pygame.display.set_caption("Freshest.AI")
screen = pygame.display.set_mode([screenWidth,screenHeight])

indexUI = 0;
font = cv2.FONT_HERSHEY_SIMPLEX

lastUIDisplayTime = 0
currentUI = None

 # Load Tensorflow model
with tf.gfile.Open('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:

    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    group = pygame.sprite.Group()
    try:
        while True:

            if (time.time() - lastUIDisplayTime > 0.5):
                group.empty();
                currentUI = None

            ret, frame = camera.read()

            rows = frame.shape[0]
            cols = frame.shape[1]

            inp = cv2.resize(frame, (cols, rows))

            # Run Tensorflow model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            num_detections = int(out[0][0])
            filteredIndex = set()
            classDelectedList = []

            # Filter sorted detected item index by score
            for i in range(num_detections):

                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                category = getCategory(classId)

                # if a classId already been detected threshold is higher
                threshold = 0.1 + classDelectedList.count(classId) * 0.1

                if score > threshold and category != '':
                    filteredIndex.add(i)
                    classDelectedList.append(classId)

                    if len(filteredIndex) > 5:
                        break
            
            # Display bounding box and UI
            first = True
            for i in filteredIndex:

                classId = int(out[3][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                category = getCategory(classId)

                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv2.rectangle(inp, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)
            
                if first:
                    if (currentUI == None):
                        image = pygame.image.load(getImagePath(classId, indexUI+1))
                        currentUI = pygame.sprite.Sprite()
                        currentUI.image = image
                        currentUI.rect = image.get_rect()
    
                        group.add(currentUI)

                    currentUI.rect.x = max(0, min(x, screenWidth - currentUI.rect.width))
                    currentUI.rect.y = max(0, min(y, screenHeight - currentUI.rect.height))
                    group.update()

                    lastUIDisplayTime = time.time()
                else:
                    cv2.putText(inp, category, (int(x), int(y)), font, 0.5, (255,255,255), 2, cv2.LINE_AA)   

                first = False
                  

            screen.fill([0,0,0])
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            inp = np.rot90(inp)
            inp = cv2.flip(inp,0)
            inp = pygame.surfarray.make_surface(inp)

            screen.blit(inp, (0,0))

            if (currentUI != None):
                group.draw(screen)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    sys.exit(0)
                if event.type == pygame.MOUSEBUTTONUP:
                    indexUI = (indexUI + 1) % 4

    except KeyboardInterrupt:
        pygame.quit()
        cv2.destroyAllWindows()
