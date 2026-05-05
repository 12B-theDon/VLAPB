# VLAPB MoveIt Teleoperation/Planning Guide

This folder contains the MoveIt-side workflow used to convert VLAPB/LIBERO
suite episodes into Panda pick-and-place planning tasks, plan waypoint
trajectories with MoveIt 2, and optionally replay one trajectory through
`ros2_control` for RViz inspection.

The tested environment for this project is:

- Ubuntu host with Docker and X11 forwarding
- ROS 2 Rolling inside the MoveIt container
- MoveIt 2 workspace installed at `/root/ws_moveit/install`
- VLAPB checkout mounted/available at `/home/artemis/Documents/VLAPB`
- Panda arm controllers:
  - `/panda_arm_controller/follow_joint_trajectory`
  - `/panda_hand_controller/gripper_cmd`

## Install Python Extras

Most ROS packages come from the MoveIt container or apt. The small Python
requirements file in this directory is for non-ROS helper scripts.

```bash
cd /home/artemis/Documents/VLAPB/libero/teleop_moveit
python3 -m pip install -r requirements.txt
```

If you are using a Conda environment outside the ROS container, install the
same file there too before generating suite data.

## Start the MoveIt Container

From this directory:

```bash
cd /home/artemis/Documents/VLAPB/libero/teleop_moveit
DOCKER_IMAGE=main-rolling-tutorial-source docker compose run cpu
```

For an NVIDIA GPU machine:

```bash
cd /home/artemis/Documents/VLAPB/libero/teleop_moveit
DOCKER_IMAGE=main-rolling-tutorial-source docker compose run gpu
```

Inside the container, source ROS and the MoveIt workspace:

```bash
source /opt/ros/rolling/setup.bash
source /root/ws_moveit/install/setup.bash
```

## Launch MoveIt/RViz

In one terminal inside the container, launch the MoveIt demo stack:

```bash
source /opt/ros/rolling/setup.bash
source /root/ws_moveit/install/setup.bash

ros2 launch moveit2_tutorials mtc_demo.launch.py
```

Check that controllers are active:

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
```

Expected controllers:

```text
joint_state_broadcaster  active
panda_hand_controller    active
panda_arm_controller     active
```

If a controller is missing or inactive, the robot can plan but replay will not
move the RViz robot.

## Generate MoveIt Task Specs

Generate `mtc_task.json` files from VLAPB suite BDDL files. Start with a small
limit to confirm the environment:

```bash
source /opt/ros/rolling/setup.bash
source /root/ws_moveit/install/setup.bash

python3 /home/artemis/Documents/VLAPB/libero/scripts/generate_suite_moveit_data.py \
  --suite belongings \
  --backend ros2-mtc-spec \
  --pose-source bddl \
  --output-dir /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit \
  --limit 100 \
  --continue-on-error
```

Generate all suites:

```bash
source /opt/ros/rolling/setup.bash
source /root/ws_moveit/install/setup.bash

python3 /home/artemis/Documents/VLAPB/libero/scripts/generate_suite_moveit_data.py \
  --all \
  --backend ros2-mtc-spec \
  --pose-source bddl \
  --output-dir /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit \
  --continue-on-error
```

Generate one known episode:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/generate_suite_moveit_data.py \
  --suite belongings \
  --episode belongings_type1_000001 \
  --backend ros2-mtc-spec \
  --pose-source bddl \
  --output-dir /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit
```

The batch planner discovers episode directories named `ep_*`. If a generated
episode has a descriptive name, create a symlink for local testing:

```bash
cd /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit
ln -sfn belongings_type1_000001 ep_000001
```

## Batch Plan Waypoint Trajectories

Run planning for one episode:

```bash
source /opt/ros/rolling/setup.bash
source /root/ws_moveit/install/setup.bash

cd /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit

python3 /home/artemis/Documents/VLAPB/libero/scripts/run_moveit_planning_batch.py \
  --episode ep_000001 \
  --all-steps \
  --continue-on-error \
  --timeout 60 \
  --heartbeat 10 \
  --failure-tail-lines 20 \
  --summary-file /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit/planning_summary.json \
  --summary-every 25 \
  --planner-mode waypoint \
  --use-waypoint-collision-scene \
  --waypoint-pick-x-offset 0.16 \
  --waypoint-max-pick-world-x 0.78 \
  --waypoint-pick-z-offset 0.09 \
  --waypoint-hover-z-offset 0.30 \
  --waypoint-pre-grasp-extra-z-offset 0.12 \
  --waypoint-place-z-offset 0.22 \
  --waypoint-post-release-lift-z-offset 0.34 \
  --target-grasp-collision-scale 0.35
```

Run planning for every generated `ep_*` directory:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/run_moveit_planning_batch.py \
  --all-steps \
  --skip-existing \
  --continue-on-error \
  --timeout 60 \
  --heartbeat 10 \
  --failure-tail-lines 20 \
  --summary-file /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit/planning_summary.json \
  --summary-every 25 \
  --planner-mode waypoint \
  --use-waypoint-collision-scene \
  --waypoint-pick-x-offset 0.16 \
  --waypoint-max-pick-world-x 0.78 \
  --waypoint-pick-z-offset 0.09 \
  --waypoint-hover-z-offset 0.30 \
  --waypoint-pre-grasp-extra-z-offset 0.12 \
  --waypoint-place-z-offset 0.22 \
  --waypoint-post-release-lift-z-offset 0.34 \
  --target-grasp-collision-scale 0.35
```

Successful planning writes files like:

```text
moveit/ep_000001/step_0_trajectory.json
```

The batch runner also writes:

```text
moveit/planning_summary.json
moveit/planning_failures.json
```

`planning_summary.json` contains overall success/failure counts, runtime
statistics, failure-stage counts, trajectory point/duration statistics, and
`outlier_trajectories` for steps that are unusually short or long. It is
checkpointed every 25 attempted jobs in the commands above, and it is also
written when the batch is interrupted with Ctrl-C. The checkpoint interval and
outlier thresholds can be overridden:

```bash
--summary-every 25
--min-trajectory-points 50
--max-trajectory-points 2000
--min-trajectory-duration 5.0
--max-trajectory-duration 180.0
```

## Replay One Planned Trajectory

The waypoint planner writes JSON trajectories. To actually move the RViz/fake
controller robot, replay the JSON:

```bash
source /opt/ros/rolling/setup.bash
source /root/ws_moveit/install/setup.bash

python3 /home/artemis/Documents/VLAPB/libero/scripts/replay_moveit_waypoint_trajectory.py \
  /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit/belongings_type1_000001/step_0_trajectory.json \
  --action /panda_arm_controller/follow_joint_trajectory \
  --gripper-action /panda_hand_controller/gripper_cmd \
  --object-id cookies_1 \
  --sync-task-objects \
  --attach-existing-world-object \
  --speed-scale 4.0
```

Use `--debug-scene` when debugging object generation, attachment, or object
counts:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/replay_moveit_waypoint_trajectory.py \
  /home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit/belongings_type1_000001/step_0_trajectory.json \
  --action /panda_arm_controller/follow_joint_trajectory \
  --gripper-action /panda_hand_controller/gripper_cmd \
  --object-id cookies_1 \
  --sync-task-objects \
  --attach-existing-world-object \
  --debug-scene \
  --speed-scale 4.0
```

The important debug lines are:

```text
debug scene after fixture sync: world=6 attached=0
debug scene after attach: world=5 attached=1
debug scene after detach: world=6 attached=0
```

For the sample belongings episode, `world=6` means four movable objects, one
basket fixture, and the table. During grasp, `cookies_1` should disappear from
world objects and appear as an attached object on `panda_hand`.

## Notes and Limitations

- `execute:=true` in `vlapb_mtc_runner` is not enough for waypoint mode in this
  setup; use `replay_moveit_waypoint_trajectory.py` to move the fake
  `ros2_control` robot.
- MoveIt/RViz collision objects are boxes, not full LIBERO meshes. The basket
  therefore appears as a collision box unless a separate mesh/marker visualizer
  is added.
- RViz/fake controllers do not simulate physical grasp. The replay script makes
  grasp visible by removing the target collision object from the world and
  attaching it to `panda_hand`.
- If the target appears duplicated, run replay with `--sync-task-objects`; it
  removes stale `vlapb_object_*` objects and recreates clean task objects from
  `mtc_task.json`.
- If planning says `No MoveIt planning jobs found`, check that the output
  directory contains `ep_*` episode directories with `mtc_task.json` files.

## Runtime Estimate

On the current environment, one waypoint planning job usually takes about
5-8 seconds. The full generated suite set is large, so plan a small subset first
and then launch the full batch with `--skip-existing`.
