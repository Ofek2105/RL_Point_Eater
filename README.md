![image](https://github.com/user-attachments/assets/eabbc57d-f2d2-42ad-baa3-1486bbb6102d)This project Attempts to Train a DQN model to get as many points as possible in a screen.
 
The state is constracted like so:

1. player x
2. player y
3. player direction, x intensity
4. player direction, y intensity
the dots are sorted by euclidian closest to player.
for each dot we have:
6. dot_x - player_x
7. dot_y - player_y
8. bollean: 0 if taken, else 1

then if max_dots is 20, the state site will be:
state_size = 4 + 3*20 = 64

the availible action are (turn left, turn right, do nothing)

this is a snapshop of the screen:

![image](https://github.com/user-attachments/assets/6c01c2d6-328c-47c7-b87c-2402b003ca36)
