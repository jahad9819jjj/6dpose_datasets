import numpy as np
import open3d as o3d
import polyscope as ps
import argparse
import os
import natsort
import pickle
from dataclasses import dataclass

@dataclass
class ObjectAnnotation:
    model: o3d.geometry.TriangleMesh = None
    class_id: int = None
    instance_id: int = None
    rotation: np.ndarray = None
    translation: np.ndarray = None
    scale: np.ndarray = None

def read_pcd(img_path, depth_path, intrinsics_path):
    img_o3d = o3d.io.read_image(img_path)
    depth_o3d = o3d.io.read_image(depth_path)
    intrinsics = np.array(np.loadtxt(intrinsics_path))
    camera_param_o3d = o3d.camera.PinholeCameraIntrinsic()
    height = np.asarray(img_o3d).shape[0]
    width = np.asarray(img_o3d).shape[1]
    camera_param_o3d.set_intrinsics(
        width, height, intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    )
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(img_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
    pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, camera_param_o3d)
    return pcd_o3d
    
def get_annotation(annotation_path, obj_models_path):
    with open(annotation_path, 'rb') as f:
        annotations_info = pickle.load(f)
    obj_anns = []
    for idx, model in enumerate(annotations_info['model_list']):
        model_path = os.path.join(obj_models_path, model.split('-')[0], model + '.obj')
        print(model_path)
        obj_ann = ObjectAnnotation()
        obj_ann.model = o3d.io.read_triangle_mesh(model_path)
        obj_ann.class_id = annotations_info['class_ids'][idx]
        obj_ann.instance_id = annotations_info['instance_ids'][idx]
        obj_ann.rotation = annotations_info['rotations'][idx]
        obj_ann.translation = annotations_info['translations'][idx]
        obj_ann.scale = annotations_info['scales'][idx]
        obj_anns.append(obj_ann)
    return obj_anns

def show_annotation(pcd, annotations):
    # TODO: Increase visualizability
    # ps.init()
    # ps_cloud = ps.register_point_cloud("scene", np.asarray(pcd.points), enabled=True)
    # ps_cloud.add_color_quantity("scene_color", np.asarray(pcd.colors))
    # ps.show()
    
    crd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    show_3d = [pcd, crd]
    for obj_ann in annotations:
        obj_ann.model.translate(obj_ann.translation)
        obj_ann.model.rotate(obj_ann.rotation)
        show_3d.append(obj_ann.model)
    
    o3d.visualization.draw_geometries(show_3d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', type=str, 
         default='/Volumes/SSD_USB/Dataset/Housecat6d/scene1-10/scene02'
    )
    parser.add_argument(
        '--object_models_path', type=str,
        default='/Volumes/SSD_USB/Dataset/Housecat6d/obj_models_small_size_final'
    )
    args = parser.parse_args()
    
    rgb_files = natsort.natsorted([os.path.join(args.dataset_path, 'rgb', f) for f in os.listdir(os.path.join(args.dataset_path, 'rgb'))])
    depth_files = natsort.natsorted([os.path.join(args.dataset_path, 'depth', f) for f in os.listdir(os.path.join(args.dataset_path, 'depth'))])
    intrinsic_files = os.path.join(args.dataset_path, 'intrinsics.txt')
    
    pcd = read_pcd(rgb_files[0], depth_files[0], intrinsic_files)
    
    annotations_files = natsort.natsorted([os.path.join(args.dataset_path, 'labels', f) for f in os.listdir(os.path.join(args.dataset_path, 'labels'))])
    annotations = get_annotation(annotations_files[0], args.object_models_path)
    show_annotation(pcd, annotations)
    
    
    