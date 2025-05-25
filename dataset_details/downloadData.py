import kagglehub

# Download latest version
path = kagglehub.dataset_download("coder98/emotionpain")

print("Path to dataset files:", path)