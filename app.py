import streamlit as st
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d as conv
import matplotlib.pyplot as plt
from time import time


def main_layout(img_dir):

    # ---------- Main Layout ---------------

    st.title("SYDE 673: Optical Flow")

    st.write("""
        # Instructions:
        Select the Optical Flow method from the side panel on the left. Choose the parameters for that method.
        Then select the images to run it on. Click RUN to display the results.
    """)

    st.write("""## Which images would you like to use?""")
    option1 = "Use images from KITTI dataset"
    option2 = "Upload my own images"
    radio = st.radio("Select one", [option1, option2])

    if radio == option1:
        img_idx = str(
            st.number_input(
                "Choose the image number between 0 and 199",
                min_value=0,
                max_value=199,
                value=0,
            )
        )
        img_files = [
            img_dir + "0" * (6 - len(img_idx)) + img_idx + "_10.png",
            img_dir + "0" * (6 - len(img_idx)) + img_idx + "_11.png",
        ]
    elif radio == option2:
        img_files = st.file_uploader(
            "Must upload two image files to run", accept_multiple_files=True
        )

    st.write(
        """**Note:** First image will be considered at time t and second image at time t+1"""
    )

    btn = False
    if img_files:
        for img_file in img_files:
            st.image(img_file)

        if len(img_files) == 2:
            btn = st.button("RUN")

    return btn, img_files


def sidebar_layout():

    # -------------- Side bar layout -----------------------

    st.sidebar.write("""
        # Select the Optical Flow method and required parameters
    """)
    method = st.sidebar.selectbox(
        "Select Optical Flow Method",
        ["Lucas Kanade", "Horn Schunk"],
    )

    if method == "Lucas Kanade":
        winSize = st.sidebar.slider(
            "Select Window Size", min_value=3, max_value=20, value=5)
        tau = st.sidebar.select_slider("Select Eigen Value threshold",
                                       [0.001, 0.01, 0.1, 1.0], value=0.01)
        params = {
            "winSize": winSize,
            "tau": tau,
        }
    if method == "Horn Schunk":
        alpha = st.sidebar.slider(
            "Select the regularization parameter for brightness constancy", min_value=0.0, max_value=10.0, value=1.0)
        Niter = st.sidebar.select_slider("Select the number of iterations to run the optimizer for",
                                         [50, 100, 150, 200], value=100)
        params = {
            "alpha": alpha,
            "Niter": Niter
        }

    return method, params


def lucas_kanade(images, window_size=5, tau=1e-2):
    """
    Inputs:
    images: list of 2 images at time t and t+1
    window_size: patch size of (window_size x window_size) around each pixel
    tau: threshold for smallest eigen value to ensure "cornerness" or validity of flow

    Output:
    (u,v, valid) = a tuple of u and v components of the flow field at each point and whether the point is valid (0 or 1)
    """

    img1 = images[0]
    img2 = images[1]

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25

    # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    w = int(window_size/2)

    # ------Implement Lucas Kanade--------

    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = conv(img1, kernel_x, boundary='symm', mode=mode)
    fy = conv(img1, kernel_y, boundary='symm', mode=mode)
    ft = conv(img2, kernel_t, boundary='symm', mode=mode) + \
        conv(img1, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    valid = np.zeros(img1.shape)
    # within window window_size * window_size
    for i in range(w, img1.shape[0]-w):
        for j in range(w, img1.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here
            # if the smallest eigenvalue of A'A is larger than the threshold τ:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]
                valid[i, j] = 1

    return (u, v, valid)


def horn_schunk(images, alpha=1, Niter=100):
    """
    Inputs:
    images: list of 2 images at time t and t+1
    alpha: regularization constant
    Niter: number of iteration

    Output:
    (u,v) = a tuple of u and v components of the flow field at each point and whether the point is valid (0 or 1)
    """

    img1 = images[0]
    img2 = images[1]

    # derivative kernels
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]]) * .25
    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,    0, 1/6],
                       [1/12, 1/6, 1/12]], float)

    # set up initial velocities
    u = np.zeros([img1.shape[0], img1.shape[1]])
    v = np.zeros([img1.shape[0], img1.shape[1]])

    mode = "same"
    # Estimate derivatives
    fx = conv(img1, kernel_x, boundary='symm', mode=mode)
    fy = conv(img1, kernel_y, boundary='symm', mode=mode)
    ft = conv(img2, kernel_t, boundary='symm', mode=mode) + \
        conv(img1, -kernel_t, boundary='symm', mode=mode)

    # ------- Implement Horn-Schunk ---------------

    # Iteration to reduce error
    for _ in range(Niter):
        # Compute local averages of the flow vectors
        uAvg = conv(u, kernel, boundary='symm', mode=mode)
        vAvg = conv(v, kernel, boundary='symm', mode=mode)
        # common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        # iterative step
        u = uAvg - fx * der
        v = vAvg - fy * der

    # All points are valid since it's a dense field
    valid = np.ones(img1.shape)

    return (u, v, valid)


def visualize_flow(u, v, im1):
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32)
    hsv[..., 1] = 1
    mag, ang = cv.cartToPolar(u, v)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(rgb)
    ax1.axis("off")

    x, y = np.where(np.abs(mag) > 50)

    r, c = im1.shape[0], im1.shape[1]
    cols, rows = np.meshgrid(np.arange(0, c, 1), np.arange(r, -1, -1))

    ax2.imshow(im1, cmap='gray')
    t = 10
    ax2.quiver(cols[x, y][::5], rows[x, y][::5], u[x, y][::5], v[x, y][::5],
               color="r", angles='xy', units='xy', width=0.5)
    ax2.axis("off")

    return fig


def main(img_dir):
    btn, img_files = main_layout(img_dir)
    method, params = sidebar_layout()

    images = []
    for img_file in img_files:

        if isinstance(img_file, st.runtime.uploaded_file_manager.UploadedFile):
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            images.append(image)
        else:
            image = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
            images.append(image)

    if btn:
        start_time = time()
        with st.spinner("Running " + method + ". Will take a few minutes..."):
            if method == "Lucas Kanade":
                u, v, _ = lucas_kanade(
                    images, params["winSize"], params["tau"])
            if method == "Horn Schunk":
                u, v, _ = horn_schunk(images, params["alpha"], params["Niter"])

            run_time = time() - start_time
            st.success(
                f"Done! Now try a different combination of method, parameters and images! Run time = {run_time: .2f}s")

        fig = visualize_flow(u, v, images[0])
        st.pyplot(fig)

        st.write("""
                    # A note on visualization:
                    In the first image, to visualize the flow field, the direction of motion is set to hue and magnitude is set to saturation using the HSV color representation, 
                    with all values being set to a maximum of 255. The HSV is then transformed to RGB and displayed above.

                    In the second image, a quiver plot is overlaid on the image at time t. In order to avoid overcrowding of the vectors, they are first thresholded at a magnitude 
                    of 200 and then only every 5th vector is considered for plotting.
                    """)


if __name__ == "__main__":

    # Set the KITTI image_2 directory path
    img_dir = "./image_2/"

    main(img_dir)
