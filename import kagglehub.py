import kagglehub

# Download latest version
path = kagglehub.dataset_download("debayank2024/netflix-movies-and-series")

print("Path to dataset files:", path)