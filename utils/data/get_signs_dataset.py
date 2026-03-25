import kagglehub

# Download latest version
path = kagglehub.dataset_download("valentynsichkar/traffic-signs-preprocessed")

print("Path to dataset files:", path)
