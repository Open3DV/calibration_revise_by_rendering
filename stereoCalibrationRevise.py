import cv2
import os
import numpy as np

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
    true_pts_list_l = []
    for idx in range(image_nums):
        render_path = 'temp_files/render_result/%02d_L.png'%idx
        img = cv2.imread(render_path)

        is_found, center_pts = cv2.findCirclesGrid(cv2.bitwise_not(img), patternSize=(9, 13), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        true_pts, _ = cv2.projectPoints(obj_pts, rvecs_l[idx], tvecs_l[idx], camera_matrix_1, dist_coeffs_1)
        true_pts_list_l.append(true_pts)
        for i in range(center_pts.shape[0]):
            true_pt = true_pts[i][0]
            center_pt = center_pts[i][0]
            scaled_true_pt = (true_pt-center_pt)*500 + center_pt
            center_pt = np.array(center_pt).astype(np.int32)
            scaled_true_pt = np.array(scaled_true_pt).astype(np.int32)
            cv2.circle(img, center_pt, 3, (0, 0, 255), -1)
            cv2.line(img, center_pt, scaled_true_pt, (0, 255, 0), 1)
        cv2.imwrite('temp_files/show_true_point/%02d_L.png'%idx, img)

    true_pts_list_r = []
    for idx in range(image_nums):
        render_path = 'temp_files/render_result/%02d_R.png'%idx
        img = cv2.imread(render_path)
        is_found, center_pts = cv2.findCirclesGrid(cv2.bitwise_not(img), patternSize=(9, 13), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        true_pts, _ = cv2.projectPoints(obj_pts, rvecs_r[idx], tvecs_r[idx], camera_matrix_2, dist_coeffs_2)
        true_pts_list_r.append(true_pts)
        for i in range(center_pts.shape[0]):
            true_pt = true_pts[i][0]
            center_pt = center_pts[i][0]
            scaled_true_pt = (true_pt-center_pt)*500 + center_pt
            center_pt = np.array(center_pt).astype(np.int32)
            scaled_true_pt = np.array(scaled_true_pt).astype(np.int32)
            cv2.circle(img, center_pt, 3, (0, 0, 255), -1)
            cv2.line(img, center_pt, scaled_true_pt, (0, 255, 0), 1)
        cv2.imwrite('temp_files/show_true_point/%02d_R.png'%idx, img)

    # 双目标定
    print('obj_pts_list:', len(obj_pts_list))
    print('true_pts_list_l:', len(true_pts_list_l))
    print('true_pts_list_r:', len(true_pts_list_r))

    (
        re_projection_err,
        camera_matrix_1,
        dist_coeffs_1,
        rvecs_l,
        tvecs_l,
    ) = cv2.calibrateCamera(
        objectPoints=obj_pts_list,
        imagePoints=true_pts_list_l,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    print("camera_matrix_r", camera_matrix_1)
    print("dist_coeffs_r", dist_coeffs_1)

    (
        re_projection_err,
        camera_matrix_2,
        dist_coeffs_2,
        rvecs_r,
        tvecs_r,
    ) = cv2.calibrateCamera(
        objectPoints=obj_pts_list,
        imagePoints=true_pts_list_r,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    print("camera_matrix_r", camera_matrix_2)
    print("dist_coeffs_r", dist_coeffs_2)

    re_projection_err, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=obj_pts_list,
        imagePoints1=true_pts_list_l,
        imagePoints2=true_pts_list_r,
        cameraMatrix1=camera_matrix_1,
        distCoeffs1=dist_coeffs_1,
        cameraMatrix2=camera_matrix_2,
        distCoeffs2=dist_coeffs_2,
        imageSize=image_size,
        R=R,
        T=T,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

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
            scaled_true_pt = (true_pt-center_pt)*1 + center_pt
            center_pt = np.array(center_pt).astype(np.int32)
            scaled_true_pt = np.array(scaled_true_pt).astype(np.int32)
            cv2.circle(image, center_pt, 1, (0, 0, 255), -1)
            cv2.line(image, center_pt, scaled_true_pt, (0, 255, 0), 1)
    cv2.imshow('reprojection error', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return 