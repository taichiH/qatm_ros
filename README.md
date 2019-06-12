# qatm_ros
ros package for template matching using QATM

## QATM implementation
`qatm.py` and `qatm_pytorch.py` are mostly copied from https://github.com/cplusx/QATM . 

I have to say thank you for @cplusx.

## Instalation
```
git clone https://github.com/taichiH/qatm_ros
catkin build qatm_ros
```

## Dependencies
This package depend on `jsk_recognition_msgs` to output rect and labels. \
```
roscd qatm_ros
rosdep install -y -r --from-paths --ignore-src .
```

## Run
default
```
roslaunch qatm_ros qatm.launch
```

If you don't have a gpu.
```
roslaunch qatm_ros qatm.launch use_cuda:=false
```

## Input and Output
### Subscribing Topic
`~input` (`sensor_msgs/Image`)

### Publishing Topic
`~output` (`sensor_msgs/Image`) \
`~output/labels` (`jsk_recognition_msgs/LabelArray`) \
`~output/rects` (`jsk_recognition_msgs/RectArray`) 

### Parameters
`use_cuda` (Boolean, default:`true`) \
`threshold` (Double, default:`0.95`)
`templates_dir` (String, default:$(find qatm_ros)/templates) \
`alpha` (Int, default:`25`)

## Create original templates
create template image file named like this package sample templates and put files on templates directory.\```
```
original_templates/template-1.png
original_templates/template-2.png
```
change template_dir param on launch file
```
<arg name="templates_dir" default="$(find qatm_ros)/original_templates" />
```
or
```
roslaunch qatm_ros qatm.launch templates_dir:=${PKG_PATH}/original_templates
```