# GP Validation Simulation with R1 rover


```
make px4_sitl gz_r1_rover
```

```
micro-xrce-dds-agent udp4 -p 8888
```

```
ros2 launch control rover_bringup.launch.py 
```

```
ros2 run control position_controller.py 
```
Enter co-ordinates like "10 10" (without quotes)

```
ros2 run control temp_query.py 
```
Press Enter to view temperature at rover's location in the field.

