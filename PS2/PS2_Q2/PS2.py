import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

def divergence(px,py):
    dpx_dx = np.gradient(px, axis=1)  # x = columns
    dpy_dy = np.gradient(py, axis=0)  # y = rows

    return dpx_dx + dpy_dy
def interpolation(v,phi):
    # v ◦ phi
    vx_phi = map_coordinates(
        v[0],
        [phi[1], phi[0]],
        order=3,
        mode='nearest'
    )

    vy_phi = map_coordinates(
        v[1],
        [phi[1], phi[0]],
        order=3,
        mode='nearest'
    )

    dphi_dt = np.stack([vx_phi, vy_phi])
    return dphi_dt

# integration_step
h = 0.1
def eq(v0,t,phi0):
    v = v0.copy()
    phis = [phi0]

    for i in range(t):
        v = v + h * euler(v)
        phis.append(h * interpolation(v, phis[-1]) + phis[-1])

    return v, phis
def deq(v0,t,phi0):
    v = v0.copy()
    phis_inv = [phi0]
    for i in range(t):
        v = v + h * -1 * euler(v)
        phix_inv = phis_inv[-1][0]
        phiy_inv = phis_inv[-1][1]
        # D(phi^-1)
        phix_x = np.gradient(phix_inv, axis=1)
        phix_y = np.gradient(phix_inv, axis=0)

        phiy_x = np.gradient(phiy_inv, axis=1)
        phiy_y = np.gradient(phiy_inv, axis=0)

        # D(phi^-1) * v
        Dphi_inv_v = np.stack([
            phix_x * v[0] + phix_y * v[1],
            phiy_x * v[0] + phiy_y * v[1]
        ])
        phis_inv.append(phis_inv[-1]- (h * Dphi_inv_v ))
    return v,phis_inv

def euler(v):
    vx = v[0]
    vy = v[1]

    vxx = np.gradient(vx, axis=1)
    vxy = np.gradient(vx, axis=0)

    vyx = np.gradient(vy, axis=1)
    vyy = np.gradient(vy, axis=0)

    # Jacobian Term
    Dv = np.array([[vxx, vxy], [vyx, vyy]])

    Dv_v = np.stack([
        vxx * vx + vxy * vy,
        vyx * vx + vyy * vy
    ])

    DvT_v = np.stack([
        vxx * vx + vyx * vy,
        vxy * vx + vyy * vy
    ])

    return -K(DvT_v + Dv_v + v * divergence(vx, vy))

from scipy.ndimage import gaussian_filter, map_coordinates


def K(field, sigma=1):
    smoothed = np.zeros_like(field)

    smoothed[0] = gaussian_filter(field[0], sigma=sigma)
    smoothed[1] = gaussian_filter(field[1], sigma=sigma)

    return smoothed
def deform_source(source, phi):
    return map_coordinates(
        source,
        [phi[1], phi[0]],
        order=3,
        mode='nearest'
    )
def generate_velocity(source, mu=0, sigma=1, smooth_sigma=3):
    # ∇s
    ds_dy, ds_dx = np.gradient(source)

    # ε ~ N(mu, sigma^2)
    epsilon = np.random.normal(
        mu, sigma, source.shape
    )

    # Smooth ε with Gaussian kernel
    epsilon = gaussian_filter(epsilon, sigma=smooth_sigma)

    # v0 = ε * ∇s
    vx = epsilon * ds_dx
    vy = epsilon * ds_dy

    velocity = np.stack([vx, vy])

    return velocity
if __name__ == "__main__":

    # Read in data as a 2, 100, 100 vector field'''
    velocity = torch.load('/Users/yuezhangchen/Documents/CS/ML/Image Analysis/PS2/PS2_Q2/code+data/v0.pt').numpy()
    print("Dimension of velocity V0: ", velocity.shape)

    # Read in data as a 1, 100, 100  image
    source = torch.load('/Users/yuezhangchen/Documents/CS/ML/Image Analysis/PS2/PS2_Q2/code+data/source.pt').numpy()
    print("Dimension of image: ", source.shape)

    # image = Image.open("/Users/yuezhangchen/Documents/CS/ML/Image Analysis/PS2/PS2_Q2/code+data/bison.webp").convert("L")
    # image = image.resize((100, 100))
    # source = np.asarray(image, dtype=np.float64) / 255.0
    # velocity = generate_velocity(source,mu=7,sigma=1)
    # print("Dimension of velocity V0: ", velocity.shape)
    # print("Dimension of image: ", source.shape)

    t0 = time.time()

    # the initial condition for ϕ0 is an image coordinate of the given source image
    H, W = source.shape
    y, x = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing='ij'
    )
    phi0 = np.stack([x, y]).astype(float)

    v, phis = deq(velocity, 10, phi0)
    print("Velocity: ", v)
    # for phi in phis:
    #     plt.imshow(deform_source(source,phi), cmap='gray')
    #     plt.show()

    fig, axes = plt.subplots(1, len(phis), figsize=(20, 2))

    for i, phi in enumerate(phis):
        axes[i].imshow(deform_source(source, phi), cmap='gray')
        axes[i].set_title(f"{i * h:.1f}")

        # axes[i].axis('off')

    # plt.tight_layout()
    plt.show()
    t = time.time() - t0
    print("Time: ", t)