import numpy as np
import matplotlib.pyplot as plt
import random


# apply gamma correction to convert sRGB to linear RGB
def gamma_correction(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# reverse gamma correction to convert linear RGB back to sRGB
def linear_to_gamma(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) - 0.055)


# convert linear RGB to XYZ color space
def rgb_to_xyz(rgb):
    rgb = gamma_correction(rgb)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    return np.dot(rgb, M.T)


# convert XYZ to IPT color space
def xyz_to_ipt(xyz):
    M_xyz_to_lms = np.array([
        [0.4002, 0.7075, -0.0808],
        [-0.2263, 1.1653, 0.0457],
        [0.0000, 0.0000, 0.9182]
    ])
    lms = np.dot(xyz, M_xyz_to_lms.T)
    lms = np.sign(lms) * np.abs(lms) ** 0.43  # apply non-linear response
    M_lms_to_ipt = np.array([
        [0.4000, 0.4000, 0.2000],
        [4.4550, -4.8510, 0.3960],
        [0.8056, 0.3572, -1.1628]
    ])
    return np.dot(lms, M_lms_to_ipt.T)


# revert IPT to XYZ color space
def ipt_to_xyz(ipt):
    M_ipt_to_lms = np.array([
        [1.0000, 0.0976, 0.2052],
        [1.0000, -0.1139, 0.1332],
        [1.0000, 0.0326, -0.6769]
    ])
    lms = np.dot(ipt, M_ipt_to_lms.T)
    lms = np.sign(lms) * np.abs(lms) ** (1 / 0.43)
    M_lms_to_xyz = np.array([
        [1.8502, -1.1383, 0.2384],
        [0.3668, 0.6439, -0.0107],
        [0.0000, 0.0000, 1.0889]
    ])
    return np.dot(lms, M_lms_to_xyz.T)


# convert linear RGB to Oklab color space using provided formula
def linear_srgb_to_oklab(c):
    l = 0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]
    m = 0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]
    s = 0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    return np.array([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    ])


# revert Oklab to linear RGB color space using provided formula
def oklab_to_linear_srgb(c):
    l_ = c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2]
    m_ = c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2]
    s_ = c[0] - 0.0894841775 * c[1] - 1.2914855480 * c[2]

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    return np.array([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    ])


# convert XYZ to RGB color space
def xyz_to_rgb(xyz):
    M = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    rgb = np.dot(xyz, M.T)
    return linear_to_gamma(rgb)


# generate a linear gradient between two colors
def generate_gradient(start_rgb, end_rgb, q):
    return np.linspace(start_rgb, end_rgb, q)


# plot the gradient in different color spaces
def plot_gradients(start_rgb, end_rgb, q):
    start_rgb = np.array(start_rgb) / 255.0
    end_rgb = np.array(end_rgb) / 255.0

    # generate the gradient in the rgb space
    gradient_rgb = generate_gradient(start_rgb, end_rgb, q)

    # convert the gradient to xyz space
    gradient_xyz = rgb_to_xyz(gradient_rgb)

    # convert to ipt and oklab, then back to rgb for display
    gradient_ipt = np.array([xyz_to_rgb(ipt_to_xyz(c)) for c in
                             np.linspace(xyz_to_ipt(rgb_to_xyz(start_rgb)), xyz_to_ipt(rgb_to_xyz(end_rgb)), q)])
    gradient_oklab = np.array([linear_to_gamma(oklab_to_linear_srgb(c)) for c in
                               np.linspace(linear_srgb_to_oklab(gamma_correction(start_rgb)),
                                           linear_srgb_to_oklab(gamma_correction(end_rgb)), q)])

    # set up the plot
    fig, axs = plt.subplots(3, 1, figsize=(6, 4))

    # plot rgb gradient
    axs[0].imshow([gradient_rgb], aspect="auto")
    axs[0].set_title("sRGB")
    axs[0].axis("off")

    # plot ipt gradient
    axs[1].imshow([gradient_ipt], aspect="auto")
    axs[1].set_title("IPT")
    axs[1].axis("off")

    # plot oklab gradient
    axs[2].imshow([gradient_oklab], aspect="auto")
    axs[2].set_title("Oklab")
    axs[2].axis("off")

    plt.show()


# specify start and end rgb colors, and how many steps in the gradient
start = 0
stop = 255
start_rgb = [random.randrange(start, stop), random.randrange(start, stop), random.randrange(start, stop)]
end_rgb = [random.randrange(start, stop), random.randrange(start, stop), random.randrange(start, stop)]
q = 8
print(start_rgb, end_rgb)

# plot the gradients
plot_gradients(start_rgb, end_rgb, q)
