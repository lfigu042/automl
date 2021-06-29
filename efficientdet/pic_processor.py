#print obj count, labels, bounding boxes and accuracy
import pdb
import csv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import inference
#from inference import ServingDriver
import pickle
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# if tf.executing_eagerly():
#    tf.compat.v1.disable_eager_execution()
import hparams_config
import PIL
from PIL import Image
from pathlib import Path
from visualize import vis_utils
from keras import label_util
import matplotlib.pyplot as plt

objectID = 0
image_id = 0


class Furkan:

    def __init__(self,img_id, image_number, boxes, classes, scores, labels, img, pickle_path='data.bin'):
        self.img_id = img_id
        self.image_number = image_number
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.labels = labels
        self.img = img
        self.model = self.load_model(pickle_path)


    # creating picStatsie
    def load_model(pickle_path):
        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as f:
                random_forest_model = pickle.load(f)
        else:
            #todo train
            return
        return random_forest_model

    def for_Furkan(self):
        #ymin, xmin, ymax, xmax
      print("for_Furkan running")
      i = 0
      global objectID

      id_count = 0
      dict = {}
      field_name = ["image_name", "image_id", "objectID", "class_number", "class_name",  "y_min", "x_min", "y_max", "x_max",  "accuracy", "distance_pred"]

      #adding to dictionary
      print("LEN of SCORES >>>> " + str(len(self.scores)))
      while i < len(self.scores):
        #   print(i)
          if self.scores[i] >= 0.6:
              y_min = self.boxes[i][0]
              x_min = self.boxes[i][1]
              y_max = self.boxes[i][2]
              x_max = self.boxes[i][3]
              certainty = self.scores[i]        #percentage of detection confidence
              class_number = self.classes[i]    #label for the type of object detected (ex '1' is the label  for 'human')
              class_name = self.labels.get(class_number)    #name of object detected (ex: 'human', 'chair', etc)

            # random forest model to predict distance based on coordinates, certainty, and class num  
              distance_pred = self.model.predict([[y_min, x_min, y_max , x_max, certainty, class_number]]) #predictors = ['y_min', 'x_min', 'y_max', 'x_max','prediction', 'class_number']
              print("DISTANCE PREDICTION >>>>>>>>>>", distance_pred)
              dictionary_entry = {
                                    "image_name": self.image_number, 
                                    "image_id": self.img_id, 
                                    "objectID": objectID, 
                                    "class_number": class_number,
                                    "class_name": class_name, 
                                    "y_min": int(y_min), 
                                    "x_min": int(x_min), 
                                    "y_max": int(y_max), 
                                    "x_max": int(x_max), 
                                    "accuracy": certainty,
                                    "distance_pred": distance_pred
                                }
              #dict.update(dictionary_entry)
              with open('out.csv', 'a') as csvfile:
                  writer = csv.DictWriter(csvfile, fieldnames = field_name)
                 # writer.writeheader()
                  writer.writerow(dictionary_entry)
                  print("writing dict to .csv = " + str(dictionary_entry))
              objectID += 1
        #   else:
        #       print("certainty score not high enough")

          i += 1

      #print statements
      print("\ndict with stuff... -> " + str(dict) + "\nObjection Detection Count: " + str(i))
      # pdb.set_trace()

      #transfering stats into CSV
      #self.toCSV(dict)
      return dict

    # transfering dictionary key/value onto CSV file
    def toCSV(self, dict):
        print("toCSV running")
        with open('out.csv', 'a', newline='') as t:
            thewriter = csv.writer(t)
            for key, value in dict.items():
                thewriter.writerow([key, value])
                print(str(key) + ",,,,,,, " + str(value))
            print("writing dict to .csv = " + str(dict))
            t.close()

class det_model:

    def __init__(self, model_file_name = 'efficientdet-d5'):
        print("det_model __init__ running")
        global image_id
        self.sess = self.load_session(model_file_name)


    def load_session(model_path):
        with tf.Session() as sess:
            tf.saved_model.load(sess, ['serve'], model_path)
            
            return sess 

    def detect_objcts(img):
        if isinstance(img, str):
            img = np.array(PIL.Image.open(f))

        detections = sess.run('detections:0', {'image_arrays:0': [img]})
        return detections

        files = list(Path('./data_pics/DataSet5').rglob('*.png'))
        params = inference.hparams_config.get_detection_config('efficientdet-d5').as_dict()
        label_map = params.get('label_map', None)

        with tf.Session() as sess:
            tf.saved_model.load(sess, ['serve'], './efficientdet-d5')
            raw_images = []
            for f in tf.io.gfile.glob('./data_pics/Dataset1/*.jpg'):
                raw_images.append(np.array(PIL.Image.open(f)))
            # driver = inference.ServingDriver('efficientdet-d5', './efficientdet-d5', min_score_thresh=0.6)
            for ind, ff in enumerate(files):
                print("INDEX >>> " + str(ind))
                img_name = os.path.basename(ff)
                detections = sess.run('detections:0', {'image_arrays:0': [raw_images[ind]]})
                up_img = inference.visualize_image_prediction(ind, img_name, raw_images[ind], detections[0], label_map=label_map, min_score_thresh=0.6)
                image_id += 1
                # pdb.set_trace()
                # saving output
                PIL.Image.fromarray(up_img).save('./data_pics/D4&5_processed/' + img_name)



def main():
    # processing data
    print("main running")
    field_name = ["image_name", "image_id", "objectID", "class_number", "class_name", "y_min", "x_min", "y_max", "x_max",  "accuracy", "distance_pred"]
    # adding to dictionary
    with open('out.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
    detect = det_model()

def main_2():
    walker = smartWalker()
    img = "./data_pics/testPics/1622212109.9949608.jpg"
    boxes, classes, scores, distances = walker.process_img(img, visualize=True)

    #give_direction(boxes, classes, scores, distances)


class smartWalker:

    def __init__(self, distance_model_path='data.bin',
                 det_model_path = './efficientdet-d5'):
        # pdb.set_trace()
        
        self.det_sess = self.load_detection_session(det_model_path)
        self.dist_model = self.load_distance_model(distance_model_path)


    def load_detection_session(self, det_model_path):
        sess = tf.Session()
        tf.saved_model.load(sess, ['serve'], det_model_path)
        return sess 

    def load_distance_model(self, dist_model_path):
        model = None
        if os.path.isfile(dist_model_path):
            with open(dist_model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            #todo train
            return model
        
        return model

    def process_img(self, img, visualize=False, min_score_thresh=0.6,
                    max_boxes_to_draw = 30, line_thickness=2, **kwargs):
        if isinstance(img, str):
            img = np.array(PIL.Image.open(img))
        else:
            img = np.array(img)

        params = inference.hparams_config.get_detection_config('efficientdet-d5').as_dict()
        label_map = params.get('label_map', None)
        
        detections = self.det_sess.run('detections:0', {'image_arrays:0': [img]})
        prediction = detections[0]
        boxes = prediction[:, 1:5]
        classes = prediction[:, 6].astype(int)
        scores = prediction[:, 5]

        label_map = label_util.get_label_map(label_map or 'coco')
        category_index = {k: {'id': k, 'name': label_map[k]} for k in label_map}

        distances = []
        for ind, [y_min, x_min, y_max, x_max] in enumerate(boxes):
            certainty = scores[ind]
            class_number = classes[ind]
            distances.append(self.dist_model.predict([[y_min, x_min, y_max , x_max, certainty, class_number]]))

        if visualize:
            processed_img = vis_utils.visualize_boxes_and_labels_on_image_array(
                                img,
                                boxes,
                                classes,
                                scores,
                                category_index,
                                distances,
                                min_score_thresh=min_score_thresh,
                                max_boxes_to_draw=max_boxes_to_draw,
                                line_thickness=line_thickness,
                                **kwargs)
            
            img_plot = plt.imshow(processed_img)
            plt.show()

        boxes_filtered = []
        classes_filtered = []
        scores_filtered = []
        distances_filtered = []

        for i in range(len(scores)):
            if scores[i] > min_score_thresh:
                 boxes_filtered.append(boxes[i])
                 classes_filtered.append(classes[i])
                 scores_filtered.append(scores[i])
                 distances_filtered.append(distances[i])

        return boxes_filtered, classes_filtered, scores_filtered, distances_filtered




if __name__ == "__main__":

    main_2()
    #main()



# image: a image with shape [H, W, C].
#     boxes: a box prediction with shape [N, 4] ordered [ymin, xmin, ymax, xmax].
#     classes: a class prediction with shape [N].
#     scores: A list of float value with shape [N].
#     label_map: a dictionary from class id to name.
#     min_score_thresh: minimal score for showing. If claass probability is below
#       this threshold, then the object will not show up.
#     max_boxes_to_draw: maximum bounding box to draw.
#     line_thickness: how thick is the bounding box line.
#     **kwargs: extra parameters.
