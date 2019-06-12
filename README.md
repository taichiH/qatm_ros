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
`templates` (String, default:qatm_ros/data/templates.csv) \
`alpha` (Int, default:`25`)