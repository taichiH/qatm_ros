#!/usr/bin/env python

'''
MIT License

Copyright (c) 2019 Hiromichi Kamata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import rospy
import rospkg
import cv_bridge
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import LabelArray, Label
from jsk_recognition_msgs.msg import RectArray, Rect

from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *

import ast
import types
import sys

pkg_dir = rospkg.RosPack().get_path('qatm_ros')
qatm_pytorch = os.path.join(pkg_dir, 'scripts/qatm_pytorch.py')

print("import qatm_pytorch.py...")
with open(qatm_pytorch) as f:
       p = ast.parse(f.read())

for node in p.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        p.body.remove(node)

module = types.ModuleType("mod")
code = compile(p, "mod.py", 'exec')
sys.modules["mod"] = module
exec(code,  module.__dict__)

from mod import *

class Qatm():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.use_cuda = rospy.get_param('~use_cuda', True)
        self.thresh = rospy.get_param('~thresh', 0.95)
        self.templates_dir = rospy.get_param('~templates_dir', os.path.join(pkg_dir, 'templates'))
        self.alpha = rospy.get_param('~alpha', 25)
        self.resize_scale = rospy.get_param('~resize_scale', 1)
        self.image_width = None
        self.image_height = None

        rospy.loginfo("define model...")
        self.model = CreateModel(
            model=models.vgg19(pretrained=True).features, alpha=self.alpha, use_cuda=self.use_cuda)
        rospy.loginfo("done define model ")

        self.labels_pub = rospy.Publisher('~output/labels', LabelArray, queue_size=1)
        self.rects_pub = rospy.Publisher('~output/rects', RectArray, queue_size=1)
        self.original_size_rects_pub = rospy.Publisher('~output/original_size_rects', RectArray, queue_size=1)
        self.pub_viz = rospy.Publisher('~output', Image, queue_size=1)

        rospy.Subscriber('~input', Image, self.callback, queue_size=1)

    def check_width(self, val):
        if val < 0:
            px = 0
        elif val > self.image_width:
            px = self.image_width
        else:
            px = val
        return px

    def check_height(self, val):
        if val < 0:
            px = 0
        elif val > self.image_height:
            px = self.image_height
        else:
            px = val
        return px

    def check_out_of_image(self, box):
        lt_x = self.check_width(box[0][0])
        lt_y = self.check_height(box[0][1])
        rb_x = self.check_width(box[1][0])
        rb_y = self.check_height(box[1][1])
        width = rb_x - lt_x
        height = rb_y - lt_y
        return lt_x, lt_y, width, height

    def callback(self, imgmsg):
        raw_image = None
        self.image_width = imgmsg.width
        self.image_height = imgmsg.height
        try:
            raw_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        except:
            rospy.logerr('failed transform image')
            return

        dataset = ImageDataset(
               pkg_dir, raw_image, self.templates_dir, thresh=self.thresh, image_name=str(imgmsg.header.stamp))

        scores, w_array, h_array, label_list = calculate_scores(self.model, dataset)
        boxes, indices = nms(scores, w_array, h_array, dataset.thresh, label_list)
        output_image = plot_results(
               dataset.image_raw, boxes, label_list, indices, show=False, save_name=None)

        labels_msg = LabelArray()
        rects_msg = RectArray()
        original_size_rects_msg = RectArray()
        for i in range(len(indices)):
            rect_msg = Rect()
            label_msg = Label()
            original_size_rect_msg = Rect()

            box = boxes[i][None, :,:][0]

            x, y, width, height = self.check_out_of_image(box)
            rect_msg.x = x
            rect_msg.y = y
            rect_msg.width = width
            rect_msg.height = height

            original_size_rect_msg.x = x * 1 / self.resize_scale
            original_size_rect_msg.y = y * 1 / self.resize_scale
            original_size_rect_msg.width = width * 1 / self.resize_scale
            original_size_rect_msg.height = height * 1 / self.resize_scale

            # rect_msg.x = box[0][0]
            # rect_msg.y = box[0][1]
            # rect_msg.width = box[1][0] - box[0][0]
            # rect_msg.height = box[1][1] - box[0][1]

            label_msg.name = label_list[indices[i]]
            rects_msg.rects.append(rect_msg)
            labels_msg.labels.append(label_msg)
            original_size_rects_msg.rects.append(original_size_rect_msg)

        rects_msg.header = imgmsg.header
        labels_msg.header = imgmsg.header
        original_size_rects_msg.header = imgmsg.header

        self.labels_pub.publish(labels_msg)
        self.rects_pub.publish(rects_msg)
        self.original_size_rects_pub.publish(original_size_rects_msg)

        msg_viz = cv_bridge.CvBridge().cv2_to_imgmsg(output_image, encoding='bgr8')
        msg_viz.header = imgmsg.header
        self.pub_viz.publish(msg_viz)


if __name__ == '__main__':
    rospy.init_node('qatm_template_matching')
    qatm = Qatm()
    rospy.spin()
