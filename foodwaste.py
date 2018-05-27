import pygame
from pygame.locals import *
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
from pprint import pprint

with open('foodDatabase.json') as f:
    database = json.load(f)

def getCategory(classId):
    if str(classId) in database:
        return database[str(classId)]['name']
    return ''

def getStorageDesc(classId):
    if str(classId) in database:
        return database[str(classId)]['storageDesc']
    return ''

def getImagePath(classId):
    if str(classId) in database:
        return database[str(classId)]['image']
    return ''

camera = cv2.VideoCapture(0)

pygame.init()
pygame.display.set_caption("Ai For Good")
screen = pygame.display.set_mode([800,600])

with tf.gfile.Open('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:

    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    group = pygame.sprite.Group()
    try:
        while True:
            group.empty();
            group.add();

            ret, frame = camera.read()

            rows = frame.shape[0]
            cols = frame.shape[1]
          #  print(rows, cols)

            inp = cv2.resize(frame, (cols, rows))

              # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):

               # print(out[3])

                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                category = getCategory(classId)
                if score > 0.1 and category != '':
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(inp, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)

                    cv2.putText(inp, category, (int(x), int(y)), font, 1, (255,255,255), 2, cv2.LINE_AA)

                    image = pygame.image.load(getImagePath(classId)))
                    sprite = pygame.sprite.Sprite()
                    sprite.image = image
                    sprite.rect = image.get_rect()
                    sprite.rect.x = x
                    sprite.rect.y = y
                    if (len(group) == 0):
                        group.add(sprite)

            screen.fill([0,0,0])
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            inp = np.rot90(inp)
            inp = cv2.flip(inp,0)
            inp = pygame.surfarray.make_surface(inp)

            screen.blit(inp, (0,0))
            if (len(group) != 0):
                group.draw(screen)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    sys.exit(0)

    except KeyboardInterrupt:
        pygame.quit()
        cv2.destroyAllWindows()
