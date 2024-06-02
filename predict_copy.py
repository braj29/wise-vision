from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
# Load VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def extract_features(img):
    img = img.resize((224, 224))  # Ensure the image is resized to the input size expected by VGG16
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()  # Flatten the features to a 1-D vector

def augment_image(img):
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

    # Generate batches of augmented images
    augmented_images = []
    for batch in datagen.flow(x, batch_size=1):
        augmented_images.append(image.array_to_img(batch[0]))
        if len(augmented_images) >= 5:  # Generate 5 augmented images
            break
    return augmented_images

def extract_features_with_augmentation(img_path):
    original_img = image.load_img(img_path)
    augmented_images = augment_image(original_img)
    
    # Extract features from the original image
    features = [extract_features(original_img)]
    
    # Extract features from augmented images
    for aug_img in augmented_images:
        features.append(extract_features(aug_img))
    
    return np.mean(features, axis=0)  # Return the average feature vector


def extract_features_with_augmentation_cp(img_path):
    pil_img = pil_img.resize((224, 224))  # (224, 224)
    
    # Convert the PIL image to a numpy array
    
    augmented_images = augment_image(pil_img)
    
    # Extract features from the original image
    features = [extract_features(augmented_images)]
    
    # Extract features from augmented images
    for aug_img in augmented_images:
        features.append(extract_features(aug_img))
    
    return np.mean(features, axis=0)  # Return the average feature vector



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
    img_path2 = "D:/Downloads/image/rose3.jpg"

    # Extract features
    features1 = extract_features_with_augmentation(img_path1)
    features2 = extract_features_with_augmentation(img_path2)

    # Compare features
    euclidean_dist, cos_sim = compare_features(features1, features2)
    print(f'Euclidean Distance: {euclidean_dist}')
    print(f'Cosine Similarity: {cos_sim}')

    # Predict similarity
    is_similar = predict_similarity(features1, features2, threshold=0.8)
    print(f'Are the images similar? {"Yes" if is_similar else "No"}')
