import gdown

file_id = "1facFRq8s8oelYUtK1fj3fcfdoWoKDBQQ"  # Replace with actual file ID
file_path = "model.pth"

gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

print("Model downloaded successfully.")