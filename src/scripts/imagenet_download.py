import kagglehub

# Download latest version
path = kagglehub.dataset_download("/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet")

print("Path to dataset files:", path)