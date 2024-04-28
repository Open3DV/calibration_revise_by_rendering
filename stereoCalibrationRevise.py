import cv2
import os
import numpy as np
from spatial_transform import multiply_transform, get_rvec


def left_RT_to_right_RT(R, T, rvecs_l, tvecs_l):
    rvecs_r = []
    tvecs_r = []
    for i in range(len(rvecs_l)):
        rvec_l = rvecs_l[i]
        tvec_l = tvecs_l[i]
        rvec_r, tvec_r = multiply_transform(get_rvec(np.matrix(R)), np.matrix(T.reshape([3,1])), np.matrix(rvec_l.reshape([3,1])), np.matrix(tvec_l.reshape([3,1])))
        rvecs_r.append(rvec_r)
        tvecs_r.append(tvec_r)
    return rvecs_r, tvecs_r

def render():
    cwd = os.getcwd()
    os.chdir('render')
    os.system('calib_board_rendering.exe')
    os.chdir(cwd)
    # print("请手动双击运行render目录下的calib_board_rendering.exe")
    # print("结束后返回此处，并按任意键继续...")
    # input()


def save_calib_param(
    camera_matrix_l,
    dist_coeffs_l,
    camera_matrix_r,
    dist_coeffs_r,
    R,
    T
):
    file = cv2.FileStorage('temp_files/calib_param.xml', cv2.FileStorage_WRITE)
    file.write('camera_intrinsic_l', camera_matrix_l)
    file.write('camera_dist_l', dist_coeffs_l)
    file.write('camera_intrinsic_r', camera_matrix_r)
    file.write('camera_dist_r', dist_coeffs_r)
    file.write('R', R)
    file.write('T', T)
    file.release

def create_folders():
    if not os.path.exists('temp_files'):
        os.makedirs('temp_files')
    if not os.path.exists('temp_files/board2cam_RT'):
        os.makedirs('temp_files/board2cam_RT')
    if not os.path.exists('temp_files/render_result'):
        os.makedirs('temp_files/render_result')
    if not os.path.exists('temp_files/show_true_point'):
        os.makedirs('temp_files/show_true_point')

def stereoCalibrationRevise(
    image_list_l, image_list_r,
    camera_matrix_1, dist_coeffs_1,
    camera_matrix_2, dist_coeffs_2,
    image_pts_list_l, image_pts_list_r,
    obj_pts,
    rvecs_l, tvecs_l, rvecs_r, tvecs_r,
    R,
    T,
    image_size,
    image_nums
):
    
    create_folders()
    save_calib_param( camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T)

    
    N_l = len(rvecs_l)
    N_r = len(rvecs_r)
    for idx in range(N_l):
        file = cv2.FileStorage('temp_files/board2cam_RT/%02d_L.xml'%idx, cv2.FileStorage_WRITE)
        file.write('cam_board2L_rvec', rvecs_l[idx])
        file.write('cam_board2L_tvec', tvecs_l[idx])
        file.release    

    for idx in range(N_r):
        file = cv2.FileStorage('temp_files/board2cam_RT/%02d_R.xml'%idx, cv2.FileStorage_WRITE)
        file.write('cam_board2R_rvec', rvecs_r[idx])
        file.write('cam_board2R_tvec', tvecs_r[idx])
        file.release

    file = cv2.FileStorage('temp_files/board2cam_RT/end.xml', cv2.FileStorage_WRITE)
    file.release

    render()

    obj_pts_list = [obj_pts for i in range(N_l)]
    #true_pts_list_l = []
    revised_pts_list_l = []
    for idx in range(image_nums):
        render_path = 'temp_files/render_result/%02d_L.png'%idx
        img = cv2.imread(render_path)

        is_found, center_pts = cv2.findCirclesGrid(cv2.bitwise_not(img), patternSize=(7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        true_pts, _ = cv2.projectPoints(obj_pts, rvecs_l[idx], tvecs_l[idx], camera_matrix_1, dist_coeffs_1)
        image_pts = image_pts_list_l[idx]
        revised_pts = []
        #true_pts_list_l.append(true_pts)
        for i in range(center_pts.shape[0]):
            true_pt = true_pts[i][0]
            center_pt = center_pts[i][0]
            image_pt = image_pts[i]
            scaled_true_pt = (true_pt-center_pt)*500 + center_pt
            revised_pt = (true_pt-center_pt) + image_pt
            revised_pts.append(revised_pt)
            center_pt = np.array(center_pt).astype(np.int32)
            scaled_true_pt = np.array(scaled_true_pt).astype(np.int32)
            cv2.circle(img, center_pt, 3, (0, 0, 255), -1)
            cv2.line(img, center_pt, scaled_true_pt, (0, 255, 0), 1)
        revised_pts_list_l.append(np.array(revised_pts))
        cv2.imwrite('temp_files/show_true_point/%02d_L.png'%idx, img)

    #true_pts_list_r = []
    revised_pts_list_r = []
    for idx in range(image_nums):
        render_path = 'temp_files/render_result/%02d_R.png'%idx
        img = cv2.imread(render_path)
        is_found, center_pts = cv2.findCirclesGrid(cv2.bitwise_not(img), patternSize=(7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        true_pts, _ = cv2.projectPoints(obj_pts, rvecs_r[idx], tvecs_r[idx], camera_matrix_2, dist_coeffs_2)
        image_pts = image_pts_list_r[idx]
        revised_pts = []
        #true_pts_list_r.append(true_pts)
        for i in range(center_pts.shape[0]):
            true_pt = true_pts[i][0]
            center_pt = center_pts[i][0]
            image_pt = image_pts[i]
            scaled_true_pt = (true_pt-center_pt)*500 + center_pt
            revised_pt = (true_pt-center_pt) + image_pt
            revised_pts.append(revised_pt)
            center_pt = np.array(center_pt).astype(np.int32)
            scaled_true_pt = np.array(scaled_true_pt).astype(np.int32)
            cv2.circle(img, center_pt, 3, (0, 0, 255), -1)
            cv2.line(img, center_pt, scaled_true_pt, (0, 255, 0), 1)
        revised_pts_list_r.append(np.array(revised_pts))
        cv2.imwrite('temp_files/show_true_point/%02d_R.png'%idx, img)

    #revised_pts_list_l = np.array(revised_pts_list_l)
    #revised_pts_list_r = np.array(revised_pts_list_r)

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
        objectPoints=obj_pts_list,
        imagePoints1=revised_pts_list_l,
        imagePoints2=revised_pts_list_r,
        cameraMatrix1=camera_matrix_1,
        distCoeffs1=dist_coeffs_1,
        cameraMatrix2=camera_matrix_2,
        distCoeffs2=dist_coeffs_2,
        imageSize=image_size,
        R=R,
        T=T,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    rvecs_r, tvecs_r = left_RT_to_right_RT(R, T, rvecs_l, tvecs_l)

    #show_reproject_error( obj_pts, revised_pts_list_l, camera_matrix_1, dist_coeffs_1, rvecs_l, tvecs_l, image_size)
    #show_reproject_error( obj_pts, revised_pts_list_r, camera_matrix_2, dist_coeffs_2, rvecs_r, tvecs_r, image_size)


    return re_projection_err, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T, E, F, rvecs_l, tvecs_l, rvecs_r, tvecs_r


def show_reproject_error(
    obj_pts,
    image_pts_list,
    camera_matrix,
    dist_coeffs,
    rvecs,
    tvecs,
    image_size
):

    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    #用点和直线画出重投影误差
    N = len(rvecs)
    for idx in range(N):
        true_pts, _ = cv2.projectPoints(obj_pts, rvecs[idx], tvecs[idx], camera_matrix, dist_coeffs)

        for i in range(len(true_pts)):
            true_pt = true_pts[i][0]
            center_pt = image_pts_list[idx][i]
            scaled_true_pt = (true_pt-center_pt)*100 + center_pt
            center_pt = np.array(center_pt).astype(np.int32)
            scaled_true_pt = np.array(scaled_true_pt).astype(np.int32)
            cv2.circle(image, center_pt, 1, (0, 0, 255), -1)
            cv2.line(image, center_pt, scaled_true_pt, (0, 255, 0), 1)
    cv2.imshow('reprojection error', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 