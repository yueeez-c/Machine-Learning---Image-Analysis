#ChatGPT was used for methods reference, debugging, setting library etc.
from filecmp import clear_cache

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

lambdas = [0.9,0.95,1,3,4,8,10]
alphas = [0.001,0.002,0.005,0.008]
variances = [0.05,0.1]
gd_tolerance = 0.0013
def frequency_smothing(img):

    #Fouriser tranform
    fft = np.fft.fft2(img)
    #Center low frequency
    fft_centered = np.fft.ifftshift(fft)

    # Keep x Frequncy
    for frequency in [7,15,31]:
        fft_shifted = keep_low_frequency(fft_centered,frequency)
        # restore_image = np.fft.ifft2(fft_shifted)

        restore_image = np.fft.ifft2(np.fft.ifftshift(fft_shifted)).real
        restore_image = np.clip(restore_image, 0, 255).astype(np.uint8)

        cv.imwrite(f"lena_{frequency}.png", restore_image)
    #keep full dimension -> original
    cv.imwrite(f"lena_full.png", img)
    # plt.imshow(np.log(1 + np.abs(fft_shifted)))
    # plt.show(np.fft.fftshift(np.fft.fft2(img)))
def keep_low_frequency(fft,frequency):
    x, y,= fft.shape

    cy, cx = x // 2, y // 2
    half = frequency // 2

    result = np.zeros_like(fft)

    result[
        cy - half: cy + half + 1,
        cx - half: cx + half + 1
    ] = fft[
        cy - half: cy + half + 1,
        cx - half: cx + half + 1
    ]

    return result
def add_gaussian_noise(image, variance):

    stddev = np.sqrt(variance)
    gaussian_noise = np.random.normal(0, stddev, image.shape)

    image = image.astype(np.float32) / 255.0

    image = (image + gaussian_noise).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32) *255
def gradient(u,f,lam):
    uy,ux = (np.gradient(u))

    # norm = np.linalg.norm(np.gradient(u))
    norm = np.sqrt(ux ** 2 + uy ** 2 + 1e-6)
    ux /= norm
    uy /= norm

    return -2 * lam * (f-u) - divergence(ux,uy)

def energy_function(f,u,lam):
    uy, ux = (np.gradient(u))
    return lam * np.sum(((f-u)**2)) +np.sum(np.sqrt(ux**2 + uy**2))

# def divergence(px,py):
#     dpx_dx = np.gradient(px, axis=1)  # x = columns
#     dpy_dy = np.gradient(py, axis=0)  # y = rows

    return dpx_dx + dpy_dy
def divergence(px,py):
    dpx_dx = np.zeros_like(px, dtype=float)
    dpy_dy = np.zeros_like(py, dtype=float)

    dpx_dx[:, 1:-1] = (px[:, 2:] - px[:, :-2]) / 2
    dpy_dy[1:-1, :] = (py[2:, :] - py[:-2, :]) / 2

    # Zero padding boundaries
    dpx_dx[:, 0] = px[:, 1] / 2
    dpx_dx[:, -1] = -px[:, -2] / 2

    dpy_dy[0, :] = py[1, :] / 2
    dpy_dy[-1, :] = -py[-2, :] / 2

    return dpx_dx + dpy_dy
def total_variation_denoise(img,lam,alpha):
    convergence = False
    f = img.copy()
    u = img.copy()
    e = []
    #gradient descent
    for i in range(500):
        g = gradient(u, f, lam)
        u_new = u - alpha * g
        change = np.max(np.abs(u_new - u))
        u = u_new
        e.append(energy_function(f,u,lam))

        if change < gd_tolerance:
            print("OUTTTTTTTTT")
            break
        print(change)

    return u,e
def mse(original, denoised):
    original = original.astype(np.float32) / 255.0
    return np.mean((original - denoised) ** 2)

if __name__ == "__main__":
    lena = cv.imread("/image/lenaNoise.png", cv.IMREAD_GRAYSCALE)
    img = cv.imread("/image/Einstein.jpeg", cv.IMREAD_GRAYSCALE)

    frequency_smothing(lena)
    # Generating different noisy images with additive Gaussian noises [0.01,0.05,0.1]

    for variance in variances:
        img2 = img.copy()
        noisy_img = add_gaussian_noise(img2, variance)
        cv.imwrite(
                    f"noisy_{variance}.png",
                    noisy_img,
                )
        best_mse = float("inf")
        best_lambda = None
        best_alpha = None
        best_image = None
        best_variance = None


        for lam in lambdas:
            for alpha in alphas:
                clean_image, e =total_variation_denoise(noisy_img.astype(np.float32) / 255.0,lam,alpha)
                current_mse = mse(img, clean_image)
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_lambda = lam
                    best_alpha = alpha
                    best_image = clean_image.copy()
                    best_variance = variance
        cv.imwrite(
            f"image_{best_variance}noise_{best_alpha}alpha_{best_lambda}lambda_{gd_tolerance}tolerance.png",
            (best_image * 255).clip(0, 255).astype(np.uint8),
        )

        plt.figure()
        plt.plot(range(len(e)), e)

        plt.xlabel("Iteration k")
        plt.ylabel("Objective E(u^k)")
        plt.title("TV Denoising Convergence")

        plt.grid(True)
        plt.show()
    # clean_image, e = total_variation_denoise(noisy_img, lam=0.1, alpha=0.001)
    # cv.imwrite(
    #     f"imea.png",
    #     (best_image * 255).clip(0, 255).astype(np.uint8),
    # )