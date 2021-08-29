# README

[Rewards parameters](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-console-train-evaluate-models.html#deepracer-reward-function-signature):
- position on track
- heading
- waypoints
- track width
- distance from center line
- all wheels on track
- speed
- steering angle

### Position on track
- `x` and `y` coordinates

### Heading
- `heading` describes orientation of vehicle in degrees
- measured CCW from x-axis
- i.e. E = 0 degrees, N = 90 deg, SW = 225 deg (or -135 deg), etc.

### Waypoints
- `waypoints` are milestone coords `[x, y]` pairs 
- `closest_waypoints` = index of the 2 closest waypoints to the vehicle
- `progress` = % of track completed
- `steps` number of steps completed

### Track width
- `track_width` = width of track in meters
- i.e. 0.60

### Distance from center line
- `distance_from_center` measures displacement from center of track
- `is_left_of_center` is a boolean

### All wheels on track
- `all_wheels_on_track` is a bool

### Speed
- `speed` measured in meters per sec

### Steering angle
- `steering_angle`
- negative if steering RIGHT
- positive if steering LEFT
