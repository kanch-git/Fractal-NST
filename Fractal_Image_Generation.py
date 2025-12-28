import numpy as np
import matplotlib.pyplot as plt
import os

IMG_SIZE = 256
NUM_IMAGES = 100

mandelbrot_dir = "mandelbrot_images"
julia_dir = "julia_images"   # ✅ FIXED

os.makedirs(mandelbrot_dir, exist_ok=True)
os.makedirs(julia_dir, exist_ok=True)

# Mandelbrot Generator
def generate_mandelbrot(xmin, xmax, ymin, ymax, img_size, max_iter):
    x = np.linspace(xmin, xmax, img_size)
    y = np.linspace(ymin, ymax, img_size)
    C = x + y[:, None] * 1j
    Z = np.zeros_like(C)
    output = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        output[mask] = i

    return output

for i in range(NUM_IMAGES):

    cx = np.random.uniform(-0.75, 0.25)
    cy = np.random.uniform(-0.5, 0.5)
    zoom = np.random.uniform(0.5, 3.0)
    scale = 1.5 / zoom

    xmin, xmax = cx - scale, cx + scale
    ymin, ymax = cy - scale, cy + scale

    max_iter = np.random.randint(100, 300)

    mandelbrot_img = generate_mandelbrot(
        xmin, xmax, ymin, ymax, IMG_SIZE, max_iter
    )

    plt.imsave(
        f"{mandelbrot_dir}/mandelbrot_{i+1}.png",
        mandelbrot_img,
        cmap=np.random.choice(["inferno", "plasma", "magma", "viridis"])
    )

print("✅ 100 diverse Mandelbrot images generated.")

# Julia Generator
def generate_julia(c, xmin, xmax, ymin, ymax, img_size, max_iter):
    x = np.linspace(xmin, xmax, img_size)
    y = np.linspace(ymin, ymax, img_size)
    Z = x + y[:, None] * 1j
    output = np.zeros(Z.shape)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + c
        output[mask] = i

    return output

for i in range(NUM_IMAGES):
    c = np.random.uniform(-0.8, 0.8) + 1j * np.random.uniform(-0.8, 0.8)

    julia_img = generate_julia(
        c,
        xmin=-1.5,
        xmax=1.5,
        ymin=-1.5,
        ymax=1.5,
        img_size=IMG_SIZE,
        max_iter=100
    )

    plt.imsave(
        f"{julia_dir}/julia_{i+1}.png",
        julia_img,
        cmap="plasma"
    )

print("100 Julia images generated.")
