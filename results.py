from webapp import lucas_kanade, horn_schunk
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def generate_results(img_dir):
    '''
    Calculates flow and saves them as uint16 .png files

    Input:
    img_dir = path to dataset
    Note: must have results folder in current directory
    '''

    for method in ["horn_schunk", "lucas_kanade"]:
        result_dir = "results/" + method + "/"

        for i in range(200):

            img_idx = str(i)
            im1 = cv.imread(img_dir + "0" * (6 - len(img_idx)) +
                            img_idx + "_10.png", cv.IMREAD_GRAYSCALE)
            im2 = cv.imread(img_dir + "0" * (6 - len(img_idx)) +
                            img_idx + "_11.png", cv.IMREAD_GRAYSCALE)

            if method == "horn_schunk":
                u, v, valid = horn_schunk([im1, im2])
            elif method == "lucas_kanade":
                u, v, valid = lucas_kanade([im1, im2])

            # Convert to unint16 based on README file
            u = (u * 64.0) + 2**15
            u = u.astype(np.uint16)

            v = (v * 64.0) + 2**15
            v = v.astype(np.uint16)

            valid = valid.astype(np.uint16)

            # in BGR format for imwrite to save it correctly
            flow = np.dstack((valid, v, u))

            cv.imwrite(result_dir + "0" * (6 - len(img_idx)) +
                       img_idx + "_10.png", flow)


def flow_read(file_directory):
    '''
    ported from provided MATLAB function to read flow images
    and convert from uint16 to float64
    '''

    # Read gt while not allowing dtype to change
    I = cv.imread(file_directory, cv.IMREAD_UNCHANGED)

    # rearrange channels to correct order for u,v and valid
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    # Convert to float64 based on provided equation
    I = I.astype(np.float64)
    F_u = (I[:, :, 0]-2**15)/64.0
    F_v = (I[:, :, 1]-2**15)/64.0
    F_valid = np.minimum(I[:, :, 2], 1)

    # Set non-valid values to zero
    F_u[F_valid == 0] = 0
    F_v[F_valid == 0] = 0

    # Create new flow array with only valid values
    F = np.dstack((F_u, F_v, F_valid))

    return F


def flow_error_map(F_gt, F_est):
    '''
    Ported from provided MATLAB function to generate the error map
    required to calculate the error
    '''

    # Seperate out channels
    F_gt_du = F_gt[:, :, 0]
    F_gt_dv = F_gt[:, :, 1]
    F_gt_val = F_gt[:, :, 2]

    F_est_du = F_est[:, :, 0]
    F_est_dv = F_est[:, :, 1]

    # Calculate error
    E_du = F_gt_du - F_est_du
    E_dv = F_gt_dv - F_est_dv
    E = np.sqrt(E_du*E_du + E_dv*E_dv)

    # Filter out non valid values
    E[F_gt_val == 0] = 0

    return E, F_gt_val


def flow_error(F_gt, F_est, pixel_range=3, percent_range=0.05):
    '''
    Ported from provided MATLAB function to calculate the flow error
    based on provided thresholds
    '''

    E, F_val = flow_error_map(F_gt, F_est)
    F_mag = np.sqrt(F_gt[:, :, 0]**2 + F_gt[:, :, 1]**2)
    n_err = np.sum(
        np.all(np.stack((F_val, E > pixel_range, E/F_mag > percent_range), axis=-1), axis=-1))
    n_total = np.sum(F_val != 0)
    f_err = n_err/n_total

    return f_err


if __name__ == "__main__":

    # Call function to generate results
    # generate_results("data_scene_flow/training/image_2/")

    # Ground truth directory
    gt_dir = "data_scene_flow/training/flow_occ/"

    for method in ["horn_schunk", "lucas_kanade"]:
        flow_error_sum = 0
        result_dir = "results/" + method + "/"

        for i in range(200):

            img_idx = str(i)

            F_gt = flow_read(gt_dir + "0" * (6 - len(img_idx)) +
                             img_idx + "_10.png")
            F_est = flow_read(result_dir + "0" * (6 - len(img_idx)) +
                              img_idx + "_10.png")

            current_flow_error = flow_error(F_gt, F_est)
            flow_error_sum += current_flow_error

        avg_flow_error = flow_error_sum / 200
        print("The average flow error over all 200 images using " +
              method + " is = " + str(avg_flow_error))
