import streamlit as st
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d as conv
import matplotlib.pyplot as plt


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


def lucas_kanade(images, window_size=15, tau=1e-2):

    I1g = images[0]
    I2g = images[1]

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    w = int(window_size/2)
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = conv(I1g, kernel_x, boundary='symm', mode=mode)
    fy = conv(I1g, kernel_y, boundary='symm', mode=mode)
    ft = conv(I2g, kernel_t, boundary='symm', mode=mode) + \
        conv(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
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

    return (u, v)


def horn_schunk(images, alpha=1, Niter=100):
    """
    Inputs:
    img_files: list of 2 images at time t and t+1
    alpha: regularization constant
    Niter: number of iteration

    Output:
    (U,V) = a tuple of u and v components of the flow field at each point
    """

    im1 = images[0]
    im2 = images[1]

    # derivative kernels
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,    0, 1/6],
                       [1/12, 1/6, 1/12]], float)

    # set up initial velocities
    U = np.zeros([im1.shape[0], im1.shape[1]])
    V = np.zeros([im1.shape[0], im1.shape[1]])

    mode = "same"
    # Estimate derivatives
    fx = conv(im1, kernel_x, boundary='symm', mode=mode)
    fy = conv(im1, kernel_y, boundary='symm', mode=mode)
    ft = conv(im2, kernel_t, boundary='symm', mode=mode) + \
        conv(im1, -kernel_t, boundary='symm', mode=mode)

    # Iteration to reduce error
    for _ in range(Niter):
        # Compute local averages of the flow vectors
        uAvg = conv(U, kernel, boundary='symm', mode=mode)
        vAvg = conv(V, kernel, boundary='symm', mode=mode)
        # common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        # iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return (U, V)


def visualize_flow(u, v):
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32)
    hsv[..., 1] = 1
    mag, ang = cv.cartToPolar(u, v)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.axis("off")

    return fig


def main(img_dir):
    btn, img_files = main_layout(img_dir)
    method, params = sidebar_layout()

    images = []
    for img_file in img_files:

        if isinstance(img_file, st.uploaded_file_manager.UploadedFile):
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            images.append(image)
        else:
            image = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
            images.append(image)

    if btn:
        with st.spinner("Running " + method + ". Will take a few minutes..."):
            if method == "Lucas Kanade":
                u, v = lucas_kanade(
                    images, params["winSize"], params["tau"])
            if method == "Horn Schunk":
                u, v = horn_schunk(images, params["alpha"], params["Niter"])

            st.success(
                "Done! Now try a different comnbination of method, parameters and images!")

        fig = visualize_flow(u, v)
        st.pyplot(fig)

        st.write("""
                    # A note on visulization:
                    To visualize the flow field, the direction of motion is set to hue and magnitude is set to saturation using the HSV color representation, with all values being set to a maxinmum of 255. The HSV is then transformed to RGB and displayed above.
                    """)


if __name__ == "__main__":

    # Set the KITTI image_2 directory path
    img_dir = "data_scene_flow/training/image_2/"

    main(img_dir)
