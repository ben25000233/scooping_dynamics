# robotic-powder-weighting

## Getting started

### To build environments
- [install pytorch](https://pytorch.org/get-started/locally/)
- [install isaacgym](https://developer.nvidia.com/isaac-gym)
- install ROS with minimum packages (for simulation)
```
pip install --extra-index-url https://rospypi.github.io/simple rospy-all
pip install --extra-index-url https://rospypi.github.io/simple rosmaster defusedxml
```
- [install ROS (for real-robot)](http://wiki.ros.org/ROS/Installation)

## Set parameters
```
Please set self.ball_amount in __init__
```
## Change tool
```
Please set self.tool in __init__
tool_type : spoon, knife, stir, fork
```
## Run
```
for data_collection
make run

for train dynamics
cd dynamics/multimodal/multimodal
python mini_main.py --config configs/training_default.yaml
```


