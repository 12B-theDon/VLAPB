#!/usr/bin/env python3
"""Replay a VLAPB waypoint trajectory JSON through ros2_control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trajectory(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if "joint_names" in payload and "points" in payload:
        return payload
    if "trajectory" in payload:
        return payload["trajectory"]
    if "joint_trajectory" in payload:
        return payload["joint_trajectory"]
    raise ValueError(f"Could not find joint trajectory in {path}")


def source_task_file(path: Path) -> Path | None:
    payload = read_json(path)
    source = payload.get("episode_source")
    if source:
        candidate = Path(str(source))
        if candidate.exists():
            return candidate
    return None


def duration_from_seconds(seconds: float) -> Duration:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int(round((seconds - whole) * 1_000_000_000)))


def make_point(raw_point: dict[str, Any], speed_scale: float, start_time: float = 0.0) -> JointTrajectoryPoint:
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in raw_point["positions"]]
    if raw_point.get("velocities") is not None:
        point.velocities = [float(value) for value in raw_point["velocities"]]
    if raw_point.get("accelerations") is not None:
        point.accelerations = [float(value) for value in raw_point["accelerations"]]
    point.time_from_start = duration_from_seconds((float(raw_point["time_from_start"]) - start_time) / speed_scale)
    return point


def group_segments(raw_points: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    segments: list[tuple[str, list[dict[str, Any]]]] = []
    for point in raw_points:
        stage_name = str(point.get("stage_name", f"segment_{point.get('segment_index', len(segments))}"))
        if not segments or segments[-1][0] != stage_name:
            segments.append((stage_name, []))
        segments[-1][1].append(point)
    return segments


def wait_future(node: rclpy.node.Node, future: Any) -> Any:
    rclpy.spin_until_future_complete(node, future)
    return future.result()


def pose_text(pose: Pose) -> str:
    return (
        f"pos=({pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f}) "
        f"quat=({pose.orientation.x:.4f}, {pose.orientation.y:.4f}, "
        f"{pose.orientation.z:.4f}, {pose.orientation.w:.4f})"
    )


def primitive_text(collision: CollisionObject) -> str:
    parts = []
    for primitive in collision.primitives:
        if primitive.type == SolidPrimitive.BOX:
            dims = ", ".join(f"{value:.4f}" for value in primitive.dimensions)
            parts.append(f"box[{dims}]")
        else:
            dims = ", ".join(f"{value:.4f}" for value in primitive.dimensions)
            parts.append(f"primitive(type={primitive.type}, dims=[{dims}])")
    if collision.meshes:
        parts.append(f"meshes={len(collision.meshes)}")
    return "; ".join(parts) if parts else "no geometry"


def collision_pose_text(collision: CollisionObject) -> str:
    object_pose = pose_text(collision.pose)
    if collision.primitive_poses:
        primitive_pose = pose_text(collision.primitive_poses[0])
        return f"object_{object_pose} primitive0_{primitive_pose}"
    return f"object_{object_pose} primitive0=<none>"


def log_task_scene(node: rclpy.node.Node, task_file: Path | None) -> None:
    if task_file is None:
        node.get_logger().info("debug task scene: no task_file found")
        return
    task = read_json(task_file)
    node.get_logger().info(f"debug task scene: {task_file}")
    for item in task.get("scene_objects", []):
        pose = pose_from_payload(item["pose"])
        node.get_logger().info(
            "debug task object "
            f"id={item.get('instance_name')} class={item.get('object_class')} "
            f"is_fixture={item.get('is_fixture', False)} {pose_text(pose)}"
        )
    for step in task.get("steps", []):
        node.get_logger().info(
            "debug task step "
            f"index={step.get('sequence_index')} object={step.get('object_name')} "
            f"receptacle={step.get('receptacle_name')} "
            f"pick={step.get('pick_pose', {}).get('position')} "
            f"place={step.get('place_pose', {}).get('position')}"
        )


def log_trajectory_summary(node: rclpy.node.Node, trajectory: dict[str, Any]) -> None:
    points = list(trajectory.get("points", []))
    node.get_logger().info(
        f"debug trajectory: joints={trajectory.get('joint_names', [])} points={len(points)}"
    )
    for stage_name, stage_points in group_segments(points):
        if not stage_points:
            continue
        start = float(stage_points[0].get("time_from_start", 0.0))
        end = float(stage_points[-1].get("time_from_start", start))
        node.get_logger().info(
            f"debug trajectory stage={stage_name} points={len(stage_points)} "
            f"time=({start:.3f}->{end:.3f})"
        )


def log_planning_scene(
    node: rclpy.node.Node,
    client: rclpy.client.Client,
    label: str,
    timeout: float,
) -> None:
    if not client.wait_for_service(timeout_sec=timeout):
        node.get_logger().warning("debug planning scene unavailable: /get_planning_scene")
        return
    request = GetPlanningScene.Request()
    request.components.components = (
        PlanningSceneComponents.WORLD_OBJECT_NAMES
        | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
    )
    response = wait_future(node, client.call_async(request))
    if response is None:
        node.get_logger().warning(f"debug scene {label}: no response")
        return
    scene = response.scene
    node.get_logger().info(
        f"debug scene {label}: world={len(scene.world.collision_objects)} "
        f"attached={len(scene.robot_state.attached_collision_objects)}"
    )
    for collision in scene.world.collision_objects:
        node.get_logger().info(
            f"debug scene world id={collision.id} frame={collision.header.frame_id} "
            f"{primitive_text(collision)} {collision_pose_text(collision)}"
        )
    for attached in scene.robot_state.attached_collision_objects:
        collision = attached.object
        node.get_logger().info(
            f"debug scene attached id={collision.id} link={attached.link_name} "
            f"frame={collision.header.frame_id} touch_links={list(attached.touch_links)} "
            f"{primitive_text(collision)} {collision_pose_text(collision)}"
        )


def apply_scene(node: rclpy.node.Node, client: rclpy.client.Client, scene: PlanningScene, label: str) -> bool:
    request = ApplyPlanningScene.Request()
    request.scene = scene
    node.get_logger().info(f"Applying planning scene: {label}")
    response = wait_future(node, client.call_async(request))
    if response is None or not response.success:
        node.get_logger().warning(f"Planning scene update failed: {label}")
        return False
    return True


def send_arm_segment(
    node: rclpy.node.Node,
    client: ActionClient,
    joint_names: list[str],
    stage_name: str,
    raw_points: list[dict[str, Any]],
    speed_scale: float,
) -> None:
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = joint_names
    start_time = float(raw_points[0]["time_from_start"])
    goal.trajectory.points = [make_point(point, speed_scale, start_time=start_time) for point in raw_points]
    final_time = goal.trajectory.points[-1].time_from_start
    node.get_logger().info(
        f"Sending arm stage {stage_name!r}: {len(goal.trajectory.points)} points over "
        f"{final_time.sec + final_time.nanosec / 1e9:.2f}s"
    )
    goal_handle = wait_future(node, client.send_goal_async(goal))
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError(f"Arm trajectory goal was rejected at stage {stage_name}")
    result = wait_future(node, goal_handle.get_result_async()).result
    if result.error_code != 0:
        raise RuntimeError(f"Arm stage {stage_name} failed: {result.error_code} {result.error_string!r}")


def send_gripper(
    node: rclpy.node.Node,
    client: ActionClient,
    position: float,
    max_effort: float,
    label: str,
) -> None:
    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = max_effort
    node.get_logger().info(f"Sending gripper {label}: position={position:.3f}")
    goal_handle = wait_future(node, client.send_goal_async(goal))
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError(f"Gripper goal was rejected while trying to {label}")
    wait_future(node, goal_handle.get_result_async())


def pose_from_payload(payload: dict[str, Any]) -> Pose:
    pose = Pose()
    position = payload.get("position", {})
    orientation = payload.get("orientation", {})
    pose.position.x = float(position.get("x", 0.0))
    pose.position.y = float(position.get("y", 0.0))
    pose.position.z = float(position.get("z", 0.0))
    pose.orientation.x = float(orientation.get("x", 0.0))
    pose.orientation.y = float(orientation.get("y", 0.0))
    pose.orientation.z = float(orientation.get("z", 0.0))
    pose.orientation.w = float(orientation.get("w", 1.0))
    return pose


def fixture_box_size(object_class: str, basket_size: list[float]) -> list[float]:
    sizes = {
        "basket": basket_size,
        "wooden_tray": [0.30, 0.22, 0.06],
        "plate": [0.22, 0.22, 0.03],
        "flat_stove": [0.34, 0.26, 0.05],
    }
    return [float(value) for value in sizes.get(object_class, [0.20, 0.16, 0.08])]


def object_box_size(object_class: str, basket_size: list[float]) -> list[float]:
    if object_class == "basket":
        return fixture_box_size(object_class, basket_size)
    sizes = {
        "alphabet_soup": [0.05, 0.05, 0.10],
        "bbq_sauce": [0.05, 0.05, 0.12],
        "butter": [0.07, 0.04, 0.04],
        "cookies": [0.06, 0.06, 0.05],
    }
    return [float(value) for value in sizes.get(object_class, [0.05, 0.05, 0.08])]


def make_task_collision_object(item: dict[str, Any], basket_size: list[float]) -> CollisionObject:
    object_class = str(item.get("object_class", ""))
    collision = CollisionObject()
    collision.header.frame_id = "world"
    collision.id = str(item["instance_name"])
    collision.operation = CollisionObject.ADD
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = object_box_size(object_class, basket_size)
    collision.primitives.append(primitive)
    collision.pose = pose_from_payload(item["pose"])
    primitive_pose = Pose()
    primitive_pose.orientation.w = 1.0
    collision.primitive_poses.append(primitive_pose)
    return collision


def sync_scene_from_task(
    node: rclpy.node.Node,
    client: rclpy.client.Client,
    task_file: Path | None,
    basket_size: list[float],
    include_objects: bool,
) -> None:
    if task_file is None:
        return
    task = read_json(task_file)
    scene = PlanningScene()
    scene.is_diff = True
    if include_objects:
        for item in task.get("scene_objects", []):
            remove = CollisionObject()
            remove.id = f"vlapb_object_{item['instance_name']}"
            remove.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(remove)
            node.get_logger().info(f"sync task object remove stale duplicate id={remove.id}")
    for item in task.get("scene_objects", []):
        is_fixture = bool(item.get("is_fixture", False))
        if not is_fixture and not include_objects:
            continue
        object_class = str(item.get("object_class", ""))
        collision = make_task_collision_object(item, basket_size)
        node.get_logger().info(
            "sync task object "
            f"id={collision.id} class={object_class} is_fixture={is_fixture} "
            f"size={collision.primitives[0].dimensions} {collision_pose_text(collision)}"
        )
        scene.world.collision_objects.append(collision)
    if scene.world.collision_objects:
        label = "task scene objects" if include_objects else "fixture object(s)"
        apply_scene(node, client, scene, f"sync {len(scene.world.collision_objects)} {label}")


def apply_attachment(
    node: rclpy.node.Node,
    client: rclpy.client.Client,
    object_id: str,
    link_name: str,
    touch_links: list[str],
    attach: bool,
    box_size: list[float],
    object_offset: list[float],
    strict: bool,
    use_existing_world_object: bool,
) -> None:
    if attach:
        node.get_logger().info(
            "attach request "
            f"object_id={object_id} link={link_name} touch_links={touch_links} "
            f"use_existing_world_object={use_existing_world_object} "
            f"box_size={box_size} offset={object_offset}"
        )
        clear_attached = PlanningScene()
        clear_attached.is_diff = True
        clear_attached.robot_state.is_diff = True
        stale_attached = AttachedCollisionObject()
        stale_attached.link_name = link_name
        stale_attached.object.id = object_id
        stale_attached.object.operation = CollisionObject.REMOVE
        clear_attached.robot_state.attached_collision_objects.append(stale_attached)
        apply_scene(node, client, clear_attached, f"clear stale attached {object_id}")

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        attached = AttachedCollisionObject()
        attached.link_name = link_name
        attached.object.id = object_id
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = touch_links
        if not use_existing_world_object:
            clear_world = PlanningScene()
            clear_world.is_diff = True
            remove_world = CollisionObject()
            remove_world.id = object_id
            remove_world.operation = CollisionObject.REMOVE
            clear_world.world.collision_objects.append(remove_world)
            apply_scene(node, client, clear_world, f"remove world object {object_id}")

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(value) for value in box_size]
            pose = Pose()
            pose.orientation.w = 1.0
            pose.position.x = float(object_offset[0])
            pose.position.y = float(object_offset[1])
            pose.position.z = float(object_offset[2])
            attached.object.header.frame_id = link_name
            attached.object.primitives.append(primitive)
            attached.object.primitive_poses.append(pose)
        else:
            node.get_logger().info(
                f"attach request uses existing world object geometry for {object_id}; "
                "attached object carries id only"
            )
        scene.robot_state.attached_collision_objects.append(attached)
        if not apply_scene(node, client, scene, f"attach {object_id} -> {link_name}"):
            message = f"Failed to attach collision object {object_id!r}; continuing arm replay"
            if strict:
                raise RuntimeError(message)
            node.get_logger().warning(message)
    else:
        node.get_logger().info(f"detach request object_id={object_id} link={link_name}")
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        attached = AttachedCollisionObject()
        attached.link_name = link_name
        attached.object.id = object_id
        attached.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects.append(attached)
        if not apply_scene(node, client, scene, f"detach {object_id} from {link_name}"):
            message = f"Failed to detach collision object {object_id!r}; continuing arm replay"
            if strict:
                raise RuntimeError(message)
            node.get_logger().warning(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument(
        "--action",
        default="/panda_arm_controller/follow_joint_trajectory",
        help="FollowJointTrajectory action name.",
    )
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Values >1 replay faster.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--gripper-action", default="/panda_hand_controller/gripper_cmd")
    parser.add_argument("--open-gripper-position", type=float, default=0.035)
    parser.add_argument("--closed-gripper-position", type=float, default=0.0)
    parser.add_argument("--gripper-max-effort", type=float, default=40.0)
    parser.add_argument("--object-id", default="cookies_1")
    parser.add_argument("--attach-link", default="panda_hand")
    parser.add_argument("--attached-box-size", nargs=3, type=float, default=[0.06, 0.06, 0.08])
    parser.add_argument("--attached-object-offset", nargs=3, type=float, default=[0.0, 0.0, 0.12])
    parser.add_argument("--basket-box-size", nargs=3, type=float, default=[0.23, 0.22, 0.13])
    parser.add_argument("--task-file", type=Path, default=None)
    parser.add_argument("--no-sync-fixtures", action="store_true")
    parser.add_argument(
        "--sync-task-objects",
        action="store_true",
        help="Add/update all task scene objects from mtc_task.json before replay, not only fixtures.",
    )
    parser.add_argument(
        "--touch-link",
        action="append",
        default=["panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger"],
        help="Touch link for the attached collision object. Can be repeated.",
    )
    parser.add_argument("--strict-attach", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--no-attach", action="store_true")
    parser.add_argument("--visual-only-arm-gripper", action="store_true")
    parser.add_argument("--attach-existing-world-object", action="store_true")
    parser.add_argument("--debug-scene", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed_scale <= 0:
        raise ValueError("--speed-scale must be positive")

    trajectory = load_trajectory(args.trajectory)
    task_file = args.task_file or source_task_file(args.trajectory)
    raw_points = list(trajectory.get("points", []))
    if not raw_points:
        raise ValueError(f"No trajectory points found in {args.trajectory}")

    rclpy.init()
    node = rclpy.create_node("vlapb_waypoint_trajectory_replayer")
    client = ActionClient(node, FollowJointTrajectory, args.action)
    gripper_client = ActionClient(node, GripperCommand, args.gripper_action)
    scene_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    get_scene_client = node.create_client(GetPlanningScene, "/get_planning_scene")

    if args.debug_scene:
        log_task_scene(node, task_file)
        log_trajectory_summary(node, trajectory)

    node.get_logger().info(f"Waiting for {args.action}")
    if not client.wait_for_server(timeout_sec=args.timeout):
        raise RuntimeError(f"Action server not available: {args.action}")
    if not args.no_gripper:
        node.get_logger().info(f"Waiting for {args.gripper_action}")
        if not gripper_client.wait_for_server(timeout_sec=args.timeout):
            raise RuntimeError(f"Action server not available: {args.gripper_action}")
    if not args.no_attach:
        node.get_logger().info("Waiting for /apply_planning_scene")
        if not scene_client.wait_for_service(timeout_sec=args.timeout):
            raise RuntimeError("Service not available: /apply_planning_scene")
        if args.debug_scene:
            log_planning_scene(node, get_scene_client, "before fixture sync", args.timeout)
        if not args.no_sync_fixtures and not args.visual_only_arm_gripper:
            sync_scene_from_task(
                node,
                scene_client,
                task_file,
                args.basket_box_size,
                include_objects=args.sync_task_objects,
            )
            if args.debug_scene:
                log_planning_scene(node, get_scene_client, "after fixture sync", args.timeout)

    joint_names = list(trajectory["joint_names"])
    for stage_name, stage_points in group_segments(raw_points):
        send_arm_segment(node, client, joint_names, stage_name, stage_points, args.speed_scale)
        if stage_name == "pick_hover":
            if not args.no_gripper:
                send_gripper(
                    node,
                    gripper_client,
                    args.closed_gripper_position,
                    args.gripper_max_effort,
                    "close",
                )
            if not args.no_attach and not args.visual_only_arm_gripper:
                if args.debug_scene:
                    log_planning_scene(node, get_scene_client, "before attach", args.timeout)
                apply_attachment(
                    node,
                    scene_client,
                    args.object_id,
                    args.attach_link,
                    args.touch_link,
                    attach=True,
                    box_size=args.attached_box_size,
                    object_offset=args.attached_object_offset,
                    strict=args.strict_attach,
                    use_existing_world_object=args.attach_existing_world_object,
                )
                if args.debug_scene:
                    log_planning_scene(node, get_scene_client, "after attach", args.timeout)
        elif stage_name == "place_hover":
            if not args.no_gripper:
                send_gripper(
                    node,
                    gripper_client,
                    args.open_gripper_position,
                    args.gripper_max_effort,
                    "open",
                )
            if not args.no_attach and not args.visual_only_arm_gripper:
                if args.debug_scene:
                    log_planning_scene(node, get_scene_client, "before detach", args.timeout)
                apply_attachment(
                    node,
                    scene_client,
                    args.object_id,
                    args.attach_link,
                    args.touch_link,
                    attach=False,
                    box_size=args.attached_box_size,
                    object_offset=args.attached_object_offset,
                    strict=args.strict_attach,
                    use_existing_world_object=args.attach_existing_world_object,
                )
                if args.debug_scene:
                    log_planning_scene(node, get_scene_client, "after detach", args.timeout)
    node.get_logger().info("Replay completed")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
