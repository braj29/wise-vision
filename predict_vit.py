import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_features_cp(pil_img: Image.Image) -> np.ndarray:
    # Preprocess the PIL image using CLIP's preprocess function
    img = preprocess(pil_img).unsqueeze(0).to(device)
    
    # Extract features using CLIP
    with torch.no_grad():
        features = model.encode_image(img)
    
    # Normalize the features
    features = features / features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy array and return as a flattened array
    return features.cpu().numpy().flatten()

def extract_features(img_path):
    # Load and preprocess the image
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    
    # Extract features using CLIP
    with torch.no_grad():
        features = model.encode_image(img)
    
    # Normalize the features
    features = features / features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy array
    return features.cpu().numpy().flatten()

def compare_features(features1, features2):
    # Cosine similarity
    cos_sim = cosine_similarity([features1], [features2])[0][0]
    
    return cos_sim

def predict_similarity(features1, features2, threshold=0.5):
    cos_sim = compare_features(features1, features2)
    similarity_score = cos_sim
    
    return similarity_score > threshold

if __name__ == '__main__':
    # Example usage
    img_path1 = 'result.jpg'
    img_path2 = 'Vochysia.jpg'

    # Extract features
    features1 = extract_features(img_path1)
    features2 = extract_features(img_path2)

    # Compare features
    cos_sim = compare_features(features1, features2)
    print(f'Cosine Similarity: {cos_sim}')

    # Predict similarity
    is_similar = predict_similarity(features1, features2, threshold=0.8)
    print(f'Are the images similar? {"Yes" if is_similar else "No"}')
