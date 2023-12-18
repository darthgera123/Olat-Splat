from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import json
import cv2
from imageio.v2 import imread,imwrite
import xml.etree.ElementTree as ET
import math
import open3d as o3d
import cv2

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def central_point(out,transform_matrix_pcl):
	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = np.array(f["transform_matrix"])[0:3,:]
		for g in out["frames"]:
			mg = g["transform_matrix"][0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			if w > 0.0001:
				totp += p*w
				totw += w
	totp /= totw
	print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		# f["transform_matrix"][0:3,3] -= totp
		f["transform_matrix"] = f["transform_matrix"].tolist()
	center_origin = np.asarray([0.5,0.1,0.5])
	transform_matrix_pcl[0:3, 3] -= center_origin[0:3]
	transform_matrix_pcl[0:3, 3] -= totp[0:3]
	return out,transform_matrix_pcl


def sharpness(imagePath):
	image = cv2.imread(imagePath)
	if image is None:
		print("Image not found:", imagePath)
		return 0
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm
def reflectZ():
	return [[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]]

def reflectY():
	return [[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]

def matrixMultiply(mat1, mat2):
	return np.array([[sum(a*b for a,b in zip(row, col)) for col in zip(*mat2)] for row in mat1])


def parse_cameras(xml_file,img_path,scale=1):
    out = dict()
    with open(xml_file, "r") as f:
            intr_data = dict()
            root = ET.parse(f).getroot()
            for idx,sensor in enumerate(root[0][0].findall('sensor')):
                id = int(sensor.get('id'))
                name = int(sensor.get('id'))
                # print(id)
                calib = sensor.find('calibration')
                w = float(calib.find('resolution').get('width'))
                h = float(calib.find('resolution').get('height'))
                fl_x = float(calib.find('f').text)/scale
                fl_y = fl_x/scale 
                cx = float(calib.find('cx').text) + w/2.0
                cy = float(calib.find('cy').text) + h/2.0
                cx = cx/scale
                cy = cy/scale
                w = w/scale
                h = h/scale
                # cx = w/2.0 
                # cy = h/2.0 
                # k1 = float(calib.find('k1').text)
                # k2 = float(calib.find('k2').text)
                # k3 = float(calib.find('k3').text)
                # p1 = float(calib.find('p1').text)
                # p2 = float(calib.find('p2').text)
                camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
                camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2
                aabb_scale = 1
                intr_data[idx] = {
                    'w':w,
                    'h':h,
                    'fl_x':fl_x,
                    'fl_y': fl_y,
                    'cx' : cx,
                    'cy' : cy,
                    # 'k1': k1, 
                    # 'k2': k2, 
                    # 'k3': k3, 
                    # 'p1': p1, 
                    # 'p2': p2, 
                    'camera_angle_x': camera_angle_x,
                    'camera_angle_y': camera_angle_y,
                    'aabb_scale': 16
                }

            chunk = root[0]  
            components = chunk.find("components")
            component_dict = {}
            if components is not None:
                for component in components:
                    transform = component.find("transform")
                    if transform is not None:
                        rotation = transform.find("rotation")
                        if rotation is None:
                            r = np.eye(3)
                        else:
                            assert isinstance(rotation.text, str)
                            r = np.array([float(x) for x in rotation.text.split()]).reshape((3, 3))
                        translation = transform.find("translation")
                        if translation is None:
                            t = np.zeros(3)
                        else:
                            assert isinstance(translation.text, str)
                            t = np.array([float(x) for x in translation.text.split()])
                        scale = transform.find("scale")
                        if scale is None:
                            s = 1.0
                        else:
                            assert isinstance(scale.text, str)
                            s = float(scale.text)

                        m = np.eye(4)
                        m[:3, :3] = r
                        m[:3, 3] = t * s
                        # m[:3, 3] = t
                        component_dict[component.get("id")] = m
            frames = list()
            for frame in root[0][2]:
                current_frame = dict()
                if not len(frame):
                    continue
                if(frame[0].tag != "transform"):
                    continue
                img_no = int(frame.get("id"))
                img_name = frame.get('label')
                # imagePath = IMGFOLDER+f"Cam_{(img_no):02d}.png"
                # imagePath = os.path.join(img_path,f"Cam_{(img_no+1):02d}")
                imagePath = os.path.join(img_path,f"Cam_{(img_no):02d}")
                # imagePath = IMGFOLDER+f"frame_{(img_no+1):05d}.png"
                # bgPath = BGFOLDER+f"Cam_{(img_no):02d}.png"
                current_frame.update({"file_path": imagePath})
                # current_frame.update({"bg_path": bgPath})
                # current_frame.update({"sharpness":sharpness(imagePath)})
                matrix_elements = [float(i) for i in frame[0].text.split()]
				
                transform_matrix = np.array([[matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]], [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]], [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]], [matrix_elements[12], matrix_elements[13], matrix_elements[14], matrix_elements[15]]])
                component_id = frame.get("component_id")
                if component_id in component_dict:
                    transform_matrix = component_dict[component_id] @ transform_matrix
                # transform = np.array([float(x) for x in camera.find("transform").text.split()]).reshape((4, 4)) 
                #swap axes
                transform_matrix = transform_matrix[[2,0,1,3],:]
                #reflect z and Y axes
                # current_frame.update({"transform_matrix":matrixMultiply(matrixMultiply(transform_matrix, reflectZ()), reflectY())} )
                current_frame.update({"transform_matrix":transform_matrix} )
                current_frame.update(intr_data[idx])
                frames.append(current_frame)
            out.update({"frames": frames})
			
    return out

       

def create_alpha(img_path,mask_path,alpha_path,scale):
	# assuming all images and mask are following same naming convention. Breaks if it doesnt
	images = sorted(os.listdir(img_path))
	N = len(images)
	for i in tqdm(range(0,N)):
		
		img = imread(os.path.join(img_path,f'Cam_{str(i).zfill(2)}.exr'))
		mask = cv2.imread(os.path.join(mask_path,f'Cam_{str(i).zfill(2)}.jpg'),cv2.IMREAD_GRAYSCALE)
		mask[mask<100] = 0
		h,w,c = img.shape
		nh,nw = int(h/scale),int(w/scale)
		nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
		nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)/255.0  
		alpha = nimg[...,:3] * (nmask)[...,np.newaxis]
		imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.exr'),alpha.astype('float32'))
		png_alpha = np.clip(np.power(np.clip(alpha,0,None)+1e-5,0.45),0,1)*255 
		imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.png'),png_alpha.astype('uint8'))

def parse_args():
	parser =  ArgumentParser(description="convert Agisoft XML export to nerf format transforms.json")

	parser.add_argument("--xml_in", default="", help="specify xml file location")
	parser.add_argument("--points_in", default="", help="specify pcl file location")
	parser.add_argument("--output_dir", default="transforms.json", help="output path")
	parser.add_argument("--points_out", default="points3d.ply", help="output path")
	parser.add_argument("--create_alpha",action='store_true', help="create_images")
	parser.add_argument("--img_path", default="images/", help="location of folder with images")
	parser.add_argument("--mask_path", default="images/", help="location of folder with images")
	parser.add_argument("--imgtype", default="png", help="type of images (ex. jpg, png, ...)")
	parser.add_argument("--scale", default=1,type=int, help="scale")
	args = parser.parse_args()
	return args



if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
	
    img_folder = os.path.join(args.output_dir,'images/')
    if args.create_alpha:
		
        os.makedirs(img_folder,exist_ok=True)
		
        create_alpha(args.img_path,args.mask_path,img_folder,args.scale)
	
    transforms = parse_cameras(args.xml_in,img_folder,args.scale)
    pcd = o3d.io.read_point_cloud(args.points_in)   
	
    
    print(pcd)

    transform_matrix_pcl = np.eye(4,dtype='float32')
    
    
    transform_matrix_pcl = transform_matrix_pcl[[2,0,1,3],:]
    
    transforms,transforms_pcd = central_point(transforms,transform_matrix_pcl)
    pcd.transform(transforms_pcd)

    print(transforms_pcd)
	
    point_cloud_out = os.path.join(args.output_dir,args.points_out)
    o3d.io.write_point_cloud(point_cloud_out, pcd)
	
    with open(os.path.join(args.output_dir,'transforms_train.json'), "w") as f:
        json.dump(transforms, f, indent=4)
		
    with open(os.path.join(args.output_dir,'transforms_test.json'), "w") as f:
        json.dump(transforms, f, indent=4)
	