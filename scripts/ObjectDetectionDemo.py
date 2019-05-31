#!/usr/bin/env python
# -*-coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CompressedImage
from robot_tensorflow.msg import detection
from robot_tensorflow.msg import objInfo
# from mymsg.msg import objInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import matplotlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetectionDemo():
    def __init__(self):
        rospy.init_node('object_detection_demo')
        rospy.on_shutdown(self.shutdown)
        # 获取launch中的参数
        model_path = rospy.get_param('~model_path',"")
        image_topic = rospy.get_param('~image_topic',"")
        # 定义需要识别的标签
        # my_object_classes = ['book','mouse','keyboard','person']
        #  申明自己的msg变量
        # detecinfo = detection()
        # objectData=objInfo()
        self._cv_bridge = CvBridge()
        # 模型位置
        PATH_TO_CKPT = '/home/feng/catkin_ws/src/robot_tensorflow/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join(model_path + '/data','mscoco_label_map.pbtxt')

        NUM_CLASSES = 90
        # 加载frozen tensorflow 模型到内存
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # 标签类别机索引值
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        #订阅image_topic话题，并调用回调函数进行识别处理
        self._sub = rospy.Subscriber(image_topic, ROSImage, self.mycallback, queue_size=1)   
        # 定义object_detection节点
        self._pub = rospy.Publisher('object_detection', ROSImage, queue_size=1)
        # 定义物体识别信息
        self.pubObjectData = rospy.Publisher('result',objInfo,queue_size=1)
        rospy.loginfo("start object dectecter")
    
    # def load_image_into_numpy_array(image):
    #     (im_width, im_height) = image.size
    #     return np.array(image.getdata()).reshape(
    #         (im_height, im_width, 3)).astype(np.uint8)
    

    def mycallback(self, image_msg):
         with self.detection_graph.as_default():
             with tf.Session(graph=self.detection_graph) as sess:
                 #使用cvbridge将ros图像数据转化为opencv图像格式
                 cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
                 pil_img = Image.fromarray(cv_image)             
                 (im_width, im_height) = pil_img.size
                #  print(pil_img.size)  640*480            
                  # the array based representation of the image will be used later in order to prepare the
                  # result image with boxes and labels on it.
                 image_np =np.array(pil_img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                 # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                 image_np_expanded = np.expand_dims(image_np, axis=0)
                 image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                 # Each box represents a part of the image where a particular object was detected.
                 boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                 # Each score represent how level of confidence for each of the objects.
                 # Score is shown on the result image, together with the class label.
                 scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                 classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                 num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                 image= Image.open
                  # Actual detection.
                 (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
             # Visualization of the results of a detection.
                 vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                 #获取位置信息
                 final_score = np.squeeze(scores)
                 count = 0
                 for i in range (100):
                     if scores is None or final_score[i]>0.5:
                         count= count+1

                # print()
                #rospy.loginfo("the count of the object is ")
                 #(im_width,im_height) = image_np.size
                 # 打印识别的物体中心点所在位置
                #  识别的bbox是[ymin,xmin,ymax,xmax]
                 for i in range (count):
                    #  print(boxes[0][i][-1])
                     y_min = boxes[0][i][0]*im_height
                     x_min = boxes[0][i][1]*im_width
                     y_max = boxes[0][i][2]*im_height
                     x_max = boxes[0][i][3]*im_width
                     object_classes = "{1}".format(i,self.category_index[classes[0][i]]['name'])
                     object_bbox= Point()
                     object_bbox.x = (x_max+x_min)/2
                     object_bbox.y = (y_max+y_min)/2
                     
                     
                     
                    #  object_classes = "{1}".format(i,self.category_index[classes[0][i]]['name'])
                    #  rospy.loginfo ("the object:%s , the position of the object is %f,%f,%f", object_classes, object_pose.x, object_pose.y,object_pose.z)
                    #  print('<----------------------------------------------------------->')
                     # 输入需要识别的物体，并发布需要抓取的物体。
                     my_object_classes = ['book','mouse','keyboard','person']
                     for object_test in my_object_classes:
                         if object_classes == object_test:
                            
                             detecinfo = detection()
                             global objectData
                             objectData = objInfo()
                             detecinfo.labels=object_classes
                             detecinfo.px = int((x_max-x_min)/2)
                             detecinfo.py = int((y_max-y_min)/2)
                             detecinfo.bbox1.x = x_min
                             detecinfo.bbox1.y = y_min
                             detecinfo.bbox2.x = x_max
                             detecinfo.bbox2.y = y_max
                             objectData.objInfos.append(detecinfo)
                            #  print (objectData.objInfos)
                             print('\n')
                             print("<-----------------------------start------------------------------>")
                             rospy.loginfo("the object <--%s--> will be grasp",object_classes)
                             print("object{0}: {1}".format(i,self.category_index[classes[0][i]]['name']),
                                   'the score of the object is :',float(scores[0][i]),
                                   ',Center_X:',detecinfo.px,'Center_Y:',detecinfo.py)
                             print (objectData.objInfos)
                            #  print(detecinfo.px)
                            #  print(detecinfo.labels)
                            #  print(detecinfo.bbox1)
                             print("<-------------------------------end---------------------------->")      
                    #发布物体检测的信息  
                     self.pubObjectData.publish(objectData)
                #  end = time.time()
                #  seconds = end - start
                #  rospy.loginfo ('time taken :{0} seconds'.format(second))
                 ros_compressed_image=self._cv_bridge.cv2_to_imgmsg(image_np,encoding="bgr8")
                 self._pub.publish(ros_compressed_image)


    def shutdown(self):
        rospy.loginfo("Stopping the tensorflow object detection...")
        rospy.sleep(1) 

if __name__ == '__main__':
    try:
        ObjectDetectionDemo()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("RosTensorFlow_ObjectDetectionDemo has started.")
