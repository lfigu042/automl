#print obj count, labels, bounding boxes and accuracy
import pdb
import csv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import inference
#from inference import ServingDriver
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

objectID = 0
image_id = 0


class Furkan:

    def __init__(self,img_id, image_number, boxes, classes, scores, labels, img):
        self.img_id = img_id
        self.image_number = image_number
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.labels = labels
        self.img = img


    # creating picStatsie

    def for_Furkan(self):
        #ymin, xmin, ymax, xmax
      print("for_Furkan running")
      i = 0
      global objectID

      id_count = 0
      dict = {}
      field_name = ["image_name", "image_id", "objectID", "class_number", "class_name",  "y_min", "x_min", "y_max", "x_max",  "accuracy"]
      #adding to dictionary
      while self.scores[i] > 0.3 and i < len(self.scores):
          print(i)
          y_min = self.boxes[i][0]
          x_min = self.boxes[i][1]
          y_max = self.boxes[i][2]
          x_max = self.boxes[i][3]
          dictionary_entry = {"image_name": self.image_number, "image_id": self.img_id, "objectID": objectID, "class_number": self.classes[i],
                              "class_name": self.labels.get(self.classes[i]), "y_min": int(y_min), "x_min": int(x_min), "y_max": int(y_max), "x_max": int(x_max), "accuracy": self.scores[i]}

          #dict.update(dictionary_entry)
          with open('pic_stat.csv', 'a') as csvfile:
              writer = csv.DictWriter(csvfile, fieldnames = field_name)
             # writer.writeheader()
              writer.writerow(dictionary_entry)
              print("writing dict to .csv = " + str(dictionary_entry))

          i += 1
          objectID += 1
      #print statements
      print("\ndict with stuff... -> " + str(dict) + "\nObjection Detection Count: " + str(i))
      # pdb.set_trace()

      #transfering stats into CSV
      #self.toCSV(dict)
      return dict

    # transfering dictionary key/value onto CSV file
    def toCSV(self, dict):
        print("toCSV running")
        with open('pic_stat.csv', 'a', newline='') as t:
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

        files = list(Path('./data_pics/testPics').rglob('*.jpg'))
        params = inference.hparams_config.get_detection_config('efficientdet-d5').as_dict()
        label_map = params.get('label_map', None)

        with tf.Session() as sess:
            tf.saved_model.load(sess, ['serve'], './efficientdet-d5')
            raw_images = []
            for f in tf.io.gfile.glob('./data_pics/testPics/*.jpg'):
                raw_images.append(np.array(PIL.Image.open(f)))
            driver = inference.ServingDriver('efficientdet-d5', './efficientdet-d5', min_score_thresh=0.3)
            for ind, ff in enumerate(files):
                print("INDEX >>> " + str(ind))
                img_name = os.path.basename(ff)
                detections = sess.run('detections:0', {'image_arrays:0': [raw_images[ind]]})
                up_img = inference.visualize_image_prediction(ind, img_name, raw_images[ind], detections[0], label_map=label_map, min_score_thresh=0.3)
                image_id += 1
                # pdb.set_trace()
                PIL.Image.fromarray(up_img).save('./data_pics/testProcessed/' + img_name)



def main():
    # processing data
    print("main running")
    field_name = ["image_name", "image_id", "objectID", "class_number", "class_name", "y_min", "x_min", "y_max", "x_max",  "accuracy"]
    # adding to dictionary
    with open('pic_stat.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
    detect = det_model()


if __name__ == "__main__":
    main()



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