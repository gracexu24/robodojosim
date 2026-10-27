"""VT-Refine compatible per-taxel SDF/virtual-spring tactile sensing.

The contact model is adapted from NVlabs/easysim-envs at commit
5512c3389685b217fae046a32b0b95f2f3020c39.  The original implementation is
Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES and MIT licensed; see
``EASYSIM_ENVS_LICENSE`` next to this file.

Unlike the original fixed plug/socket pairing, RoboDojo creates task objects
after Isaac Lab has prepared the scene.  ``VTRefineTactileBackend`` therefore
reconciles its mesh registry against the active LayoutManager records before
every observation.  Meshes are registered once, while object pose and velocity
are read live for every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    import warp as wp
except ImportError:  # Allows source/schema tooling to run outside Isaac Sim.
    wp = None


if wp is not None:

    @wp.kernel
    def _query_signed_distance(
        mesh: wp.uint64,
        query_points: wp.array(dtype=wp.vec3),
        max_distance: float,
        signed_distance: wp.array(dtype=float),
        outward_normal: wp.array(dtype=wp.vec3),
        closest_point: wp.array(dtype=wp.vec3),
    ):
        index = wp.tid()
        point = query_points[index]
        query = wp.mesh_query_point_sign_winding_number(mesh, point, max_distance)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            delta = point - closest
            distance = wp.length(delta)
            signed_distance[index] = distance * query.sign
            closest_point[index] = closest
            if distance > 1.0e-9:
                outward_normal[index] = wp.normalize(delta) * query.sign
            else:
                outward_normal[index] = wp.vec3(0.0, 0.0, 0.0)
        else:
            signed_distance[index] = max_distance
            outward_normal[index] = wp.vec3(0.0, 0.0, 0.0)
            closest_point[index] = point


@dataclass
class RegisteredObject:
    env_idx: int
    instance_name: str
    object_type: str
    prim_path: str
    obj: Any
    mesh: Any
    signature: tuple[Any, ...]
    vertex_count: int
    triangle_count: int


class VTRefineTactileBackend:
    """GPU mesh-SDF queries and VT-Refine's virtual spring contact model."""

    def __init__(
        self,
        device: str | torch.device,
        normal_stiffness: float = 1.0,
        tangential_stiffness: float = 0.1,
        friction_coefficient: float = 2.0,
        max_query_distance: float = 0.03,
    ):
        if wp is None:
            raise RuntimeError("Warp is required for the VT-Refine tactile backend")
        wp.init()
        self.device = torch.device(device)
        self.wp_device = wp.get_device(str(self.device))
        self.normal_stiffness = float(normal_stiffness)
        self.tangential_stiffness = float(tangential_stiffness)
        self.friction_coefficient = float(friction_coefficient)
        self.max_query_distance = float(max_query_distance)
        self.scene_manager = None
        self.registry: dict[tuple[int, str], RegisteredObject] = {}
        self._registration_failures: set[tuple[int, str]] = set()

    def bind_scene_manager(self, scene_manager) -> None:
        self.scene_manager = scene_manager
        self.registry.clear()
        self._registration_failures.clear()

    def reset(self) -> None:
        # Layout records can point at a different model after reset.  Reconcile
        # lazily on the next observation; Warp owns the retired mesh buffers.
        self.registry.clear()
        self._registration_failures.clear()

    @staticmethod
    def _active_objects(scene_manager, env_idx: int):
        layout_manager = scene_manager.layout_manager
        collections = {
            "Rigid": scene_manager._rigid_and_dynamic_objects,
            "Dynamic": scene_manager._rigid_and_dynamic_objects,
            "Geometry": scene_manager._geometry_objects,
        }
        for object_type, collection_by_env in collections.items():
            if env_idx >= len(collection_by_env):
                continue
            collection = collection_by_env[env_idx]
            for record in layout_manager.get_layout_records(env_idx, object_type):
                instance_name = record.get("inst_name")
                obj = collection.get(instance_name)
                if instance_name and obj is not None:
                    yield object_type, instance_name, obj, record

    def synchronize(self, env_indices) -> None:
        if self.scene_manager is None:
            raise RuntimeError("VT-Refine tactile backend is not bound to SceneManager")

        active_keys: set[tuple[int, str]] = set()
        for env_idx in env_indices:
            for object_type, instance_name, obj, record in self._active_objects(self.scene_manager, int(env_idx)):
                key = (int(env_idx), instance_name)
                active_keys.add(key)
                prim_path = getattr(obj, "usd_prim_path", getattr(obj, "_prim_path", ""))
                scale = tuple(float(value) for value in record.get("scale", getattr(obj, "scale", (1, 1, 1))))
                signature = (
                    object_type,
                    prim_path,
                    getattr(obj, "usd_path", None),
                    scale,
                    id(obj),
                )
                current = self.registry.get(key)
                if current is not None and current.signature == signature:
                    continue
                try:
                    mesh, vertex_count, triangle_count = self._create_mesh(obj, scale)
                # Spawned third-party USD assets can fail in pxr or Warp with
                # several unrelated exception types. One unsupported scene
                # prop must not disable tactile sensing for every other prop.
                except Exception as exc:  # noqa: BLE001
                    if key not in self._registration_failures:
                        print(f"[VTRefineTactile] could not register {key}: {exc}")
                        self._registration_failures.add(key)
                    continue
                self.registry[key] = RegisteredObject(
                    env_idx=int(env_idx),
                    instance_name=instance_name,
                    object_type=object_type,
                    prim_path=prim_path,
                    obj=obj,
                    mesh=mesh,
                    signature=signature,
                    vertex_count=vertex_count,
                    triangle_count=triangle_count,
                )
                self._registration_failures.discard(key)
                print(
                    f"[VTRefineTactile] registered env={env_idx} type={object_type} "
                    f"object={instance_name} vertices={vertex_count} triangles={triangle_count}"
                )

        requested_envs = {int(value) for value in env_indices}
        for key in list(self.registry):
            if key[0] in requested_envs and key not in active_keys:
                del self.registry[key]

    def _create_mesh(self, obj, scale: tuple[float, float, float]):
        from pxr import Gf, Usd, UsdGeom

        stage = getattr(obj, "stage", None)
        prim_path = getattr(obj, "usd_prim_path", getattr(obj, "_prim_path", None))
        if stage is None or not prim_path:
            raise ValueError("object has no USD stage/prim path")
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            raise ValueError(f"invalid root prim {prim_path}")

        xform_cache = UsdGeom.XformCache()
        root_world_inverse = xform_cache.GetLocalToWorldTransform(root_prim).GetInverse()
        vertices: list[list[float]] = []
        triangles: list[int] = []

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            usd_mesh = UsdGeom.Mesh(prim)
            points = usd_mesh.GetPointsAttr().Get()
            counts = usd_mesh.GetFaceVertexCountsAttr().Get()
            indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                continue
            mesh_world = xform_cache.GetLocalToWorldTransform(prim)
            offset = len(vertices)
            for point in points:
                point_world = mesh_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                point_root = root_world_inverse.Transform(point_world)
                vertices.append(
                    [
                        float(point_root[0]) * scale[0],
                        float(point_root[1]) * scale[1],
                        float(point_root[2]) * scale[2],
                    ]
                )

            cursor = 0
            for count in counts:
                count = int(count)
                face = [offset + int(index) for index in indices[cursor : cursor + count]]
                cursor += count
                for index in range(1, count - 1):
                    triangles.extend((face[0], face[index], face[index + 1]))

        if len(vertices) < 3 or len(triangles) < 3:
            raise ValueError(f"no triangulated USD meshes below {prim_path}")
        points_wp = wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3, device=self.wp_device)
        indices_wp = wp.array(np.asarray(triangles, dtype=np.int32), dtype=int, device=self.wp_device)
        mesh = wp.Mesh(points=points_wp, indices=indices_wp, support_winding_number=True)
        return mesh, len(vertices), len(triangles) // 3

    @staticmethod
    def _as_device_tensor(value, device, length: int):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.to(device=device, dtype=torch.float32).reshape(-1)[:length]

    def _object_kinematics(self, registered: RegisteredObject):
        obj = registered.obj
        if registered.object_type in ("Rigid", "Dynamic") and hasattr(obj, "get_world_pose"):
            position, orientation = obj.get_world_pose()
        elif hasattr(obj, "get_state"):
            root_pose = obj.get_state(is_relative=False)["root_pose"]
            position, orientation = root_pose[:3], root_pose[3:7]
        else:
            raise RuntimeError(f"no world-pose accessor for {registered.instance_name}")

        position = self._as_device_tensor(position, self.device, 3)
        orientation = self._as_device_tensor(orientation, self.device, 4)
        linear_velocity = torch.zeros(3, device=self.device)
        angular_velocity = torch.zeros(3, device=self.device)
        if hasattr(obj, "get_linear_velocity"):
            linear_velocity = self._as_device_tensor(obj.get_linear_velocity(), self.device, 3)
        if hasattr(obj, "get_angular_velocity"):
            angular_velocity = self._as_device_tensor(obj.get_angular_velocity(), self.device, 3)
        return position, orientation, linear_velocity, angular_velocity

    def _query_mesh(self, mesh, points_local: torch.Tensor):
        point_count = int(points_local.shape[0])
        query_points = wp.from_torch(points_local.contiguous(), dtype=wp.vec3)
        signed_distance = wp.empty(point_count, dtype=float, device=self.wp_device)
        outward_normal = wp.empty(point_count, dtype=wp.vec3, device=self.wp_device)
        closest_point = wp.empty(point_count, dtype=wp.vec3, device=self.wp_device)
        wp.launch(
            kernel=_query_signed_distance,
            dim=point_count,
            inputs=[mesh.id, query_points, self.max_query_distance],
            outputs=[signed_distance, outward_normal, closest_point],
            device=self.wp_device,
        )
        wp.synchronize_device(self.wp_device)
        return (
            wp.to_torch(signed_distance),
            wp.to_torch(outward_normal),
            wp.to_torch(closest_point),
        )

    def sample(
        self,
        env_idx: int,
        tactile_points_world: torch.Tensor,
        tactile_points_orientation_world: torch.Tensor,
        tactile_points_velocity_world: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw VT-Refine normal-force value per flattened taxel."""
        point_count = int(tactile_points_world.shape[0])
        best_distance = torch.full(
            (point_count,), self.max_query_distance, dtype=torch.float32, device=self.device
        )
        best_normal = torch.zeros((point_count, 3), dtype=torch.float32, device=self.device)
        best_closest_velocity = torch.zeros_like(best_normal)

        from isaaclab.utils.math import quat_apply, quat_conjugate

        for key, registered in self.registry.items():
            if key[0] != int(env_idx):
                continue
            position, orientation, linear_velocity, angular_velocity = self._object_kinematics(registered)
            object_orientation = orientation.unsqueeze(0).expand(point_count, -1)
            points_local = quat_apply(
                quat_conjugate(object_orientation), tactile_points_world - position.unsqueeze(0)
            )
            signed_distance, normal_local, closest_local = self._query_mesh(registered.mesh, points_local)
            take = signed_distance < best_distance
            if not torch.any(take):
                continue
            normal_world = quat_apply(object_orientation, normal_local)
            closest_offset_world = quat_apply(object_orientation, closest_local)
            closest_velocity = (
                linear_velocity.unsqueeze(0)
                + torch.linalg.cross(
                    angular_velocity.unsqueeze(0).expand(point_count, -1),
                    closest_offset_world,
                    dim=-1,
                )
            )
            best_distance = torch.where(take, signed_distance, best_distance)
            best_normal = torch.where(take.unsqueeze(-1), normal_world, best_normal)
            best_closest_velocity = torch.where(
                take.unsqueeze(-1), closest_velocity, best_closest_velocity
            )

        depth = (-best_distance).clamp(min=0.0)
        normal_force_norm = self.normal_stiffness * depth
        normal_force = normal_force_norm.unsqueeze(-1) * best_normal

        relative_velocity = tactile_points_velocity_world - best_closest_velocity
        relative_tangent = relative_velocity - best_normal * (
            best_normal * relative_velocity
        ).sum(dim=-1, keepdim=True)
        tangent_speed = torch.linalg.vector_norm(relative_tangent, dim=-1)
        static_friction = self.tangential_stiffness * tangent_speed
        dynamic_friction = self.friction_coefficient * normal_force_norm
        friction_norm = torch.minimum(static_friction, dynamic_friction)
        friction_force = (
            -friction_norm.unsqueeze(-1)
            * relative_tangent
            / tangent_speed.clamp(min=1.0e-9).unsqueeze(-1)
        )

        tactile_force_world = normal_force + friction_force
        tactile_force_local = quat_apply(
            quat_conjugate(tactile_points_orientation_world), tactile_force_world
        )
        # The ALOHA pad contact face lies on local -Y; as in VT-Refine, +Y is
        # the sensor normal axis and downstream observations clamp decompression.
        return tactile_force_local[:, 1].clamp(min=0.0)
