from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image 
from keras.applications.efficientnet import EfficientNetB0
# Load VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
# Load EfficientNetB0 model + higher level layers
# base_model = EfficientNetB0(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('top_activation').output)

def extract_features_cp(pil_img: Image.Image) -> np.ndarray:
    # Resize the image to the target size
    pil_img = pil_img.resize((224, 224))  # (224, 224)
    
    # Convert the PIL image to a numpy array
    img_data = image.img_to_array(pil_img)
    
    # Expand dimensions to match the input shape required by the model
    img_data = np.expand_dims(img_data, axis=0)
    
    # Preprocess the image data
    img_data = preprocess_input(img_data)
    
    # Predict the features using the model
    features = model.predict(img_data)
    
    # Return the features as a flattened array
    return features.flatten()

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # (224, 224)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()  # Flatten the features to a 1-D vector

def compare_features(features1, features2):
    # Euclidean distance
    euclidean_dist = euclidean(features1, features2)
    
    # Cosine similarity
    cos_sim = cosine_similarity([features1], [features2])[0][0]
    
    return euclidean_dist, cos_sim

def predict_similarity(features1, features2, threshold=0.5):
    _, cos_sim = compare_features(features1, features2)
    similarity_score = cos_sim
    # print(similarity_score)
    
    if similarity_score > threshold:
        return True
    else:
        return False



if __name__ == '__main__':
    # Example usage
    img_path1 = "D:/Downloads/image/rose.jpg"
    img_path2 = "D:/Downloads/image/rose.jpg"

    # Extract features
    features1 = extract_features(img_path1)
    features2 = extract_features(img_path2)

    # Compare features
    euclidean_dist, cos_sim = compare_features(features1, features2)
    print(f'Euclidean Distance: {euclidean_dist}')
    print(f'Cosine Similarity: {cos_sim}')

    # Predict similarity
    is_similar = predict_similarity(features1, features2, threshold=0.8)
    print(f'Are the images similar? {"Yes" if is_similar else "No"}')
