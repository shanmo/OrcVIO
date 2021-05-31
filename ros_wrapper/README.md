# Build

- Clone the repository recursively

``` shellsession
git clone --recurse-submodules <repository_url>
```

- Install dependencies and run catkin build:

``` shellsession
$ . setup_once.bash
```

Or use individual install scripts from `install-deps/` directory.


- Activate the environment

``` shellsession
$ . setup.bash
```

- Compile

``` shellsession
$ catkin build
```

# Message 

- object detection message is [wm_od_interface_msgs](https://github.com/kschmeckpeper/object_detection)
