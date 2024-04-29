import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import trimesh
import glob, tqdm, os

def render_gapart_object_with_sapien(
    urdf_paths,
    use_viewer=False,
    save_root = "sapien_render_output"
    ):
    total = len(urdf_paths)
    i = 0
    for path in tqdm.tqdm(urdf_paths, total=total):
        i += 1
        # Initialize the Sapien Engine
        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)

        # Create a scene
        scene = engine.create_scene()
        scene.set_timestep(1 / 240.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        if use_viewer:
            viewer = Viewer(renderer)
            viewer.set_scene(scene)
            viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
            viewer.set_camera_rpy(r=0, p=0, y=0)
        scene.add_ground(-5)


        # Load URDF

        loader = scene.create_urdf_loader()
        urdf_obj = loader.load(path)
        urdf_obj.set_pose(sapien.Pose([0, 0, 0], [0, 1, 0, 0]))
        # Assuming the object is static or its pose has been set already
        pose = urdf_obj.get_pose()
        print(pose)
        options = [[4, 0.00, 1, 0, -20, 0]
            , [0., 2, 1, -90, -40, 0]
            , [0., -2, 1, 90, -40, 0],
            [-4, 0.00, 1, 0, 200, 0],
            [4, 0.00, -1, 0, 40, 0]
            , [0., 2, -1, -90, 40, 0]
            , [0., -2, -1, 90, 40, 0],
            [-4, 0.00, -1, 0, -200, 0],
            [0, 0.00, 3, 0, 270, 0],
            [0, 0.00, -3, 0, 90, 0]]
        cameras = [
            scene.add_camera(
                name="camera",
                width=640,
                height=480,
                fovy=np.deg2rad(35),
                near=0.1,
                far=100,
            ) for cam in options
        ]

        for cam_id, cam_pose in enumerate(options):
            q = R.from_euler('xyz', cam_pose[3:], degrees=True).as_quat()
            cameras[cam_id].set_pose(sapien.Pose(p=cam_pose[:3], q=q))    

        scene.update_render()
        for cam in cameras: cam.take_picture() 
        positions = [cam.get_float_texture('Position') for cam in cameras]
        model_matrixs = [cam.get_model_matrix() for cam in cameras]
        points_worlds = []
        for cam_id in range(len(cameras)):
            position = positions[cam_id]
            model_matrix = model_matrixs[cam_id]
            points_opengl = position.reshape(-1, position.shape[-1])[..., :3]
            points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            points_worlds.append(points_world)
        points_worlds = np.concatenate(points_worlds, axis=0)
        valid_mask = (points_worlds[:,2]<=2.9)&(points_worlds[:,2]>=-2.9)\
        &(points_worlds[:,1]<=1.9)&(points_worlds[:,1]>=-1.9)\
        &(points_worlds[:,0]<=3.9)&(points_worlds[:,0]>=-3.9)
        points_worlds = points_worlds[valid_mask]
        rgbas = [cam.get_float_texture('Color') for cam in cameras]
        points_colors = [rgba.reshape(-1, rgba.shape[-1])[..., :3] for rgba in rgbas]
        points_colors = np.concatenate(points_colors, axis=0)[valid_mask]
        rgba_imgs = [(rgba * 255).clip(0, 255).astype("uint8") for rgba in rgbas]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_worlds)
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        
        
        os.makedirs(save_root, exist_ok=True)
        name = path.split('/')[-2]
        # import pdb; pdb.set_trace()
        print(name, i ,total)
        p_name  = path.split("/")[-1].split(".")[0]
        # import pdb; pdb.set_trace()
        o3d.io.write_point_cloud(f"{save_root}/{p_name}_{name}_point_cloud.ply", pcd)

        scene = None
        engine = None