from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from predict import extract_features, predict_similarity, compare_features, extract_features_cp
import os
import streamlit as st
model = main.deepforest()
model.use_release()

# quit()
# print(img.head())
import pandas as pd
from PIL import Image


def split_image_from_dataframe(dataframe, panoramic_image, output_folder_name):
    """
    Splits an image into multiple images based on coordinates provided in a dataframe.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing image path and coordinates.
    """
    cropped_images_info = []
    cropped_image_paths = []
    for i, row in dataframe.iterrows():
        image_path = row['image_path']
        left, top, right, bottom = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        image = Image.open(panoramic_image)
        
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image_dict = {
            f'image': cropped_image,
            f'position': (left, top, right, bottom)
        }
        cropped_images_info.append(cropped_image_dict)
        cropped_image_paths.append(f'{output_folder_name}/cropped_image_{i}.png')
        cropped_image.save(f'{output_folder_name}/cropped_image_{i}.png')  # Save each cropped image
    
    dataframe['cropped_image_path'] = cropped_image_paths
    return cropped_images_info


# print(images_list)
# quit()
# Load images from folder
def extract_treespecies_features(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', '.JPG'))]

    species_feature_list = [{"feature": extract_features(file), "file_name": file} for file in image_files]
    return species_feature_list


# print(species_feature_list[:2])
def perform_inference(images_list, species_feature_list):
    for idx, item in enumerate(images_list):
        image = item["image"]
        feature_cp = extract_features_cp(image)
        for idx, species in enumerate(species_feature_list):
            euclidean_dist, cos_sim = compare_features(feature_cp, species["feature"])
        # print(f'Euclidean Distance: {euclidean_dist}')
        # print(f'Cosine Similarity: {cos_sim}')

        # Predict similarity
            is_similar = predict_similarity(feature_cp, species["feature"], threshold=0.8)
        # print(species)
        # print(f'Are the images similar? {"Yes" if is_similar else "No"}')

            result = "Yes" if is_similar else "No"
            if result == "Yes":
                item[f"result_{idx}"] = result
                item[f"file_name_{idx}"] = species["file_name"]

    return images_list




if __name__ == '__main__':
    pan_image = "D:/Downloads/image/plant_images/plant_images/drone_igapo_flooded_forest/DJI_20240504124024_0037_D.JPG"

    sample_image_path = get_data(pan_image)
    # img = model.predict_image(path=sample_image_path, return_plot=False)
    # from PIL import Image
    # print(img)
    img_df = ""
    # img_actual = model.predict_image(path=sample_image_path, return_plot=True, color=(0, 165, 255), thickness=9)
    img_actual = model.predict_tile(raster_path=sample_image_path, return_plot=True, patch_size=100,patch_overlap=0.25)
    # im = Image.open('Foto.jpg')
    # im.save('Foto.png')
    #predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order. Many functions in deepforest will automatically perform this flip for you and give a warning.
    plt.imshow(img_actual[:,:,::-1])
    # plt.show(img[:,:,::-1])
    plt.savefig("cropped_test3/panoramic_2.png")
    quit()
    images_list = split_image_from_dataframe(img_df, pan_image)
    folder_path = 'D:/Downloads/image/plant_images/plant_images/drone_igapo_flooded_forest/identified_species'

    species_feature_list = extract_treespecies_features()
    final_result = perform_inference(images_list, species_feature_list)
    print(final_result)


