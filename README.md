# Dynamic Obstacle Avoidance: Interative Predition (SWTA) and Control (MPC)
The paper is available: [Not Yet]

![Example](doc/cover.png "Example")

## ROS simulation
[ROS XXX](https://github.com/XXX)

## Dependencies
### OpEn
The NMPC formulation is solved using open source implementation of PANOC, namely [OpEn](https://alphaville.github.io/optimization-engine/). Follow the [installation instructions](https://alphaville.github.io/optimization-engine/docs/installation) before proceeding. 

### CGAL depecencies
You need to do this. The CGAL folder here is just for reference.

To do triangulation [CGAL](https://www.cgal.org/) is used. Install this via:
```
sudo apt-get install -y libcgal-dev &&\
sudo apt-get install -y swig &&\
sudo apt-get install -y build-essential libssl-dev 
```
```
cd Documents/ &&\
sudo apt-get install -y wget &&\
sudo wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz &&\
sudo tar -zxvf cmake-3.20.0.tar.gz &&\
cd cmake-3.20.0 &&\
sudo ./bootstrap &&\
sudo make &&\
sudo make install &&\
sudo cmake --version 
```

To use CGAL with python the [python bindings](https://github.com/CGAL/cgal-swig-bindings) are used. Install these and move them to the correct location via the following commands. Note that these assume that this assumes that Formation-control-of-autonomous-transportation-robots is placed under ```Documents```.
```
cd Documents
git clone https://github.com/cgal/cgal-swig-bindings &&\
cd cgal-swig-bindings &&\
mkdir build/CGAL-5.0_release -p &&\
cd build/CGAL-5.0_release &&\
sudo cmake -DCGAL_DIR=/usr/lib/CGAL -DBUILD_JAVA=OFF -DPYTHON_OUTDIR_PREFIX=../../examples/python ../.. &&\
sudo make -j 4 &&\
cd ../../examples/python &&\ 
cp -r CGAL Documents/Formation-control-of-autonomous-transportation-robots/src/
```

### Others
Check "import" or "error" :) 
Sorry, I'm too lazy to do it here now.


## Algorithm 
The algorithm is explained in detail in the accompanying [paper](docs/Master%20thesis%20report.pdf).
 

## Short explanation of code 
In the files [obstacles.json](data/obstacles.json) and [map.json](data/map.json) one can change the obstacles and the road network that the codes uses. The algorithm can be run by running [main.py](src/main.py), make sure to set `build_solver=True` when running the program for the first time so that the Rust code is generated. One can set starting and ending positions of the robot in [main.py](src/main.py) which has to be Python tuples. start_master and start_slave can be any pose but should be in the proximity of a node in the road network  
```
start_master = (x,y,theta)
start_slave = (x,y,theta)
```
The master_goal and the slave_goal are split in to the three subgoals along the path that the trajectory has to pass. The first tuple has to be a node in the road network (except theta) and is the node at where the driving in formation starts. The second tuple also has to be a node in the road network and is the node where the formation ends. The tuple is where the robots should stop and they are no longer driving in formation at this point, this pose should also be in the proximity of a node in the road network. 

```
master_goal = [(x,y,theta),(x,y,theta),(x,y,theta)]
slave_goal= [(x,y,theta),(x,y,theta),(x,y,theta)]
```

To enable or disable the follower ATR the parameter ```enable_slave``` in [base.yaml](configs/base.yaml) should be put to either ```True``` or ```False```. To enable or disable the plotting of the follower the parameter ```plot_slave``` in [plot_config.yaml](configs/plot_config.yaml) should be put to either ```True``` or ```False```.

The code at the moment plans a trajectory for a horizon of 20 steps at a sampling time of 0.1 s, the NMPC has been tuned to regenerate that the trajectory at every 0.1 s and not 2.0 s. To see the performance of planning 2.0 s ahead and using that entire trajectory a flag ```self.two_seconds``` can be set to ```True``` in [trajectory_generator.py](src/trajectory_generator.py).

