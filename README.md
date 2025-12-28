
# Fractal-NST: Fractal-Based Neural Style Transfer for Image Augmentation

This repository provides the official implementation of the **Fractal-Based Neural Style Transfer (Fractal-NST)** framework for high-quality image augmentation. The proposed approach integrates fractal geometry with neural style transfer to generate diverse, domain-independent augmented images while preserving the structural and semantic characteristics of content images.

Fractal-NST employs non-semantic fractal patterns generated using Mandelbrot and Julia sets as style images. This strategy avoids class-specific bias commonly introduced by natural style images and enhances dataset diversity for robust deep learning model training.

---

## Repository Contents

### 1. Fractal_Image_Generation.py
This script generates fractal images based on classical fractal geometry.
- Implements Mandelbrot and Julia sets  
- Generates 100 fractal images  
- Produces non-semantic, self-similar, high-frequency texture patterns  
- Output images are used as style images for NST  

### 2. nst.py
This script performs neural style transfer-based image augmentation.
- Uses fractal-generated images as style inputs  
- Uses plant leaf images from the PlantVillage dataset as content inputs  
- Implements a VGG19-based neural style transfer architecture  
- Preserves disease-related structural features while improving texture diversity  
- Saves augmented outputs in the augmented_image directory  

---

## Folder Structure

Fractal-NST/
├── Fractal_Image_Generation.py  
├── nst.py  
├── fractal_images/  
│   ├── mandelbrot/  
│   └── julia/  
├── content_images/  
│   └── PlantVillage/  
├── augmented_image/  
└── README.md  

---

## Requirements

- Python 3.8 or later  
- PyTorch  
- torchvision  
- NumPy  
- OpenCV  
- Matplotlib  
- Pillow  

Install dependencies using:

pip install torch torchvision numpy opencv-python matplotlib pillow

---

## Usage

### Step 1: Generate Fractal Style Images

python Fractal_Image_Generation.py

The generated fractal images will be saved in the fractal_images directory.

### Step 2: Perform Image Augmentation Using Fractal-NST

Place PlantVillage images in the content_images directory and run:

python nst.py

Augmented images will be saved in the augmented_image directory.

---

## Dataset

Plant leaf images from the PlantVillage dataset are used as content images. Sample fractal images and augmented images generated using this framework are publicly available through a Zenodo repository referenced in the associated research paper.

---

## Reproducibility

All experiments reported in the paper were conducted using the code provided in this repository with fixed parameters and consistent preprocessing steps to ensure reproducibility.

---

## Citation

Please cite the corresponding research article and dataset if you use this code.

---

## License

This repository is intended for academic and research use only.
