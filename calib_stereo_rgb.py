import cv2
import numpy as np
import yaml
from stereoCalibrationRevise import stereoCalibrationRevise, show_reproject_error, left_RT_to_right_RT


def Rt2T(R_3x3, t_vec3):
    T_4x4 = np.eye(4, dtype="float64")
    T_4x4[0:3, 0:3] = R_3x3
    T_4x4[:3, 3] = t_vec3.ravel()
    return T_4x4

def dump_yaml(data, filename, method="w", encoding="utf-8", **kwargs):
    with open(filename, method) as f:
        if "b" in method:
            ret = yaml.dump(data=data, stream=f, encoding=encoding, **kwargs)
        else:
            ret = yaml.dump(data=data, stream=f, **kwargs)

    return ret

def read_images_from_path(path, file_name_head, file_name_tail, file_nums):
    image_list = []
    for i in range(file_nums):
        image_temp = cv2.imread(path + '/' + file_name_head + ('%d' % i) + file_name_tail)
        image_list.append(image_temp)

    return image_list

def gen_obj_pts(pattern_size, resolution, padding):
    obj_pts = []
    for row in range(pattern_size[1]):
        for col in range(pattern_size[0]):
            if row % 2 == 0:
                obj_pts.append([col * resolution + padding, row * resolution * 0.5 + padding, 0])
            else:
                obj_pts.append([col * resolution + 0.5 * resolution + padding, row * resolution * 0.5 + padding, 0])
    return np.float32(obj_pts)

def single_calibrate(image_list, obj_pts):
    global image_size
    image_pts_list = []
    obj_pts_list = []
    image_num_list = []

    for i in range(0, len(image_list)):
        img = image_list[i]
        image_size[0] = img.shape[1]
        image_size[1] = img.shape[0]
        img = cv2.bitwise_not(img)

        is_found, corners_nx1x2 = cv2.findCirclesGrid(img, patternSize=pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if not is_found:
            print("not find!")
            continue
        cv2.drawChessboardCorners(img, pattern_size, corners_nx1x2, is_found)
        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)
        cv2.waitKey(1)

        if is_found:
            image_pts_list.append(np.float32(corners_nx1x2).reshape(-1, 2))
            obj_pts_list.append(np.float32(obj_pts))
            image_num_list.append(i)

    cv2.destroyAllWindows()

    (
        re_projection_err,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
    ) = cv2.calibrateCamera(
        objectPoints=obj_pts_list,
        imagePoints=image_pts_list,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    print('err: ', re_projection_err)
    return camera_matrix, dist_coeffs, image_pts_list, obj_pts_list, image_num_list, rvecs, tvecs

img_l_path = "./240116_hk"
img_r_path = "./240116_hk"
pattern_size = (9, 13)
resolution = 80
padding = 60
image_size = [1624, 1240]
image_size_rgb = [0, 0]
image_nums = 50

image_list_l = read_images_from_path(img_l_path, 'L', '.bmp', image_nums)
image_list_r = read_images_from_path(img_r_path, 'R', '.bmp', image_nums)

obj_pts = gen_obj_pts(pattern_size, resolution, padding)

camera_matrix_l, dist_coeffs_l, image_pts_list_l, obj_pts_list_l, image_list_num_l, rvecs_l, tvecs_l = single_calibrate(image_list_l, obj_pts)
print("camera_matrix_l", camera_matrix_l)
print("dist_coeffs_l", dist_coeffs_l)

camera_matrix_r, dist_coeffs_r, image_pts_list_r, obj_pts_list_r, image_list_num_r, rvecs_r, tvecs_r = single_calibrate(image_list_r, obj_pts)
print("camera_matrix_r", camera_matrix_r)
print("dist_coeffs_r", dist_coeffs_r)

#show_reproject_error( obj_pts, image_pts_list_l, camera_matrix_l, dist_coeffs_l, rvecs_l, tvecs_l, image_size)
#show_reproject_error( obj_pts, image_pts_list_r, camera_matrix_r, dist_coeffs_r, rvecs_r, tvecs_r, image_size)

image_pts_list_l_temp = []
image_pts_list_r_temp = []
image_pts_list_rgb_temp = []
obj_pts_list_l_temp = []
obj_pts_list_r_temp = []
obj_pts_list_rgb_temp = []

for i in range(image_nums):
    if i in image_list_num_l and i in image_list_num_r:
        for j in range(len(image_list_num_l)):
            if i == image_list_num_l[j]:
                image_pts_list_l_temp.append(image_pts_list_l[j])
                obj_pts_list_l_temp.append(obj_pts_list_l[j])
                break
        for j in range(len(image_list_num_r)):
            if i == image_list_num_r[j]:
                image_pts_list_r_temp.append(image_pts_list_r[j])
                obj_pts_list_r_temp.append(obj_pts_list_r[j])
                break
image_pts_list_l = image_pts_list_l_temp
image_pts_list_r = image_pts_list_r_temp
image_pts_list_rgb = image_pts_list_rgb_temp
obj_pts_list_l = obj_pts_list_l_temp
obj_pts_list_r = obj_pts_list_r_temp
obj_pts_list_rgb = obj_pts_list_rgb_temp

assert len(image_pts_list_l) == len(image_pts_list_r), "not found equal num of boards"

(
    re_projection_err,
    camera_matrix_1,
    dist_coeffs_1,
    camera_matrix_2,
    dist_coeffs_2,
    R,
    T,
    E,
    F, 
    rvecs_l, tvecs_l, 
    perViewErrors
) = cv2.stereoCalibrateExtended(
    objectPoints=obj_pts_list_l,
    imagePoints1=image_pts_list_l,
    imagePoints2=image_pts_list_r,
    imageSize=image_size,
    cameraMatrix1=camera_matrix_l,
    distCoeffs1=dist_coeffs_l,
    cameraMatrix2=camera_matrix_r,
    distCoeffs2=dist_coeffs_r,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    R=None,
    T=None,
)

rvecs_r, tvecs_r = left_RT_to_right_RT(R, T, rvecs_l, tvecs_l)

print('stereo err: ', re_projection_err)
print('camera_matrix_l: ', camera_matrix_1)
print('dist_coeffs_l: ', dist_coeffs_1)

print('camera_matrix_r: ', camera_matrix_2)
print('dist_coeffs_r: ', dist_coeffs_2)
print('R: ', R)
print('T: ', T)


for i in range(0):
    (
        re_projection_err,
        camera_matrix_1, dist_coeffs_1,
        camera_matrix_2, dist_coeffs_2,
        R, T, E, F,
        rvecs_l, tvecs_l, rvecs_r, tvecs_r,
    ) = stereoCalibrationRevise(
        image_list_l, image_list_r,
        camera_matrix_1, dist_coeffs_1,
        camera_matrix_2, dist_coeffs_2,
        image_pts_list_l, image_pts_list_r,
        obj_pts,
        rvecs_l, tvecs_l, rvecs_r, tvecs_r,
        R, T,
        image_size,
        image_nums
    )
    print('the ', i, 'th iteration')
    print('re_projection_err: ', re_projection_err)

    stereo_res = {
        'cam1_k': camera_matrix_1.tolist(),
        'cam2_k': camera_matrix_2.tolist(),
        'dist_1': dist_coeffs_1.tolist(),
        'dist_2': dist_coeffs_2.tolist(),
        'R_l_r': R.tolist(),
        't_l_r': T.tolist(),
        'T': Rt2T(R, T).tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
    }

    dump_yaml(stereo_res, 'stereo_res_%d.yaml'%i)

print('stereo err: ', re_projection_err)
print('camera_matrix_l: ', camera_matrix_1)
print('dist_coeffs_l: ', dist_coeffs_1)

print('camera_matrix_r: ', camera_matrix_2)
print('dist_coeffs_r: ', dist_coeffs_2)
print('R: ', R)
print('T: ', T)


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
print("R1", R1)
print("R2", R2)
print("P1", P1)
print("P2", P2)
print("Q", Q)

stereo_res = {
    'cam1_k': camera_matrix_1.tolist(),
    'cam2_k': camera_matrix_2.tolist(),
    'dist_1': dist_coeffs_1.tolist(),
    'dist_2': dist_coeffs_2.tolist(),
    'R_l_r': R.tolist(),
    't_l_r': T.tolist(),
    'T': Rt2T(R, T).tolist(),
    'E': E.tolist(),
    'F': F.tolist(),}
dump_yaml(stereo_res, 'stereo_res.yaml')
