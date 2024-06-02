import streamlit as st
from PIL import Image
import os
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
# from predict import extract_features, predict_similarity, compare_features, extract_features_cp
import os, re
import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
from inference import split_image_from_dataframe
from datetime import datetime
from predict import extract_features, predict_similarity, compare_features, extract_features_cp
import cv2
from PIL import Image
import os
import numpy as np
import urllib.request
import glob

# intake library and plugin
# import intake
# from intake_zenodo_fetcher import download_zenodo_files_for_entry

# geospatial libraries
# import geopandas as gpd

# from rasterio.transform import from_origin
# import rasterio.features

# import fiona

# from shapely.geometry import shape, mapping, box
# from shapely.geometry.multipolygon import MultiPolygon

# # machine learning libraries
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultTrainer
# # define the URL to retrieve the model
# fn = 'model_final.pth'
# url = f'https://zenodo.org/record/5515408/files/{fn}?download=1'

# urllib.request.urlretrieve(url, config['model'] + '/' + fn)

# import geoviews.tile_sources as gts

# import hvplot.pandas
# import hvplot.xarray

# # hv.extension('bokeh', width=100)
# cfg = get_cfg()

# # if you want to make predictions using a CPU, run the following line. If using GPU, hash it out.
# cfg.MODEL.DEVICE='cuda'

# # model and hyperparameter selection
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# ### path to the saved pre-trained model weights
# cfg.MODEL.WEIGHTS = config['model'] + '/model_final.pth'

# # set confidence threshold at which we predict
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15

# #### Settings for predictions using detectron config

# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], scale=1.5, instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# image = cv2.cvtColor(v.get_image()[:, :, :], cv2.COLOR_BGR2RGB)
# st.image(image, caption='Segmented Panoramic Image Detecttree', channels ='RGB', use_column_width=True)


model = main.deepforest()
model.use_release()

# Set the page configuration
st.set_page_config(page_title="Wise-Vision", page_icon=":deciduous_tree:")

# Title and description
st.title("🌳 Wise-Vision")
st.subheader("AI + Environment Hackathon 2024")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is designed for the AI + Environment Hackathon 2024.
    Upload a panoramic image and specify a folder path to detect tree species in the image.
    Upload a word file to integrate knowledge into the image.
    Output will be a panoramic image with identified trees and knowledge symbols.
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    For more information, contact us at:
    [rajbhalwankar@protonmail.com]
    """
)


script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a new folder within the script directory for storing cropped images
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder_name = f"output_{timestamp}"
output_image_folder = os.path.join(script_dir, output_folder_name)
os.makedirs(output_image_folder, exist_ok=True)
output_image_folder = os.path.abspath(output_image_folder)
# Define paths for the image and Excel file within the new folder
cropped_image_path = os.path.join(output_image_folder, f"panoramic_{timestamp}.png")
excel_output_path = os.path.join(output_image_folder, f"results_{timestamp}.xlsx")

# Input: Upload panoramic image
uploaded_image = st.file_uploader("Upload a panoramic image", type=['png', 'jpeg', 'JPG'])

# Input: Folder path for tree species detection

def extract_treespecies_features(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', '.JPG'))]

    species_feature_list = [{"feature": extract_features(file), "file_name": file} for file in image_files]
    return species_feature_list


# print(species_feature_list[:2])
def perform_inference(cropped_images, species_feature_list, img_df):

    for img_idx, item in enumerate(cropped_images):
        image = item["image"]
        feature_cp = extract_features_cp(image)
        row_results = []
        species_result = []
        emoji = []
        species_context = []
        for idx, species in enumerate(species_feature_list):
            # euclidean_dist, cos_sim = compare_features(feature_cp, species["feature"])
        # print(f'Euclidean Distance: {euclidean_dist}')
        # print(f'Cosine Similarity: {cos_sim}')

        # Predict similarity
            is_similar = predict_similarity(feature_cp, species["feature"], threshold=0.92)
        # print(species)
        # print(f'Are the images similar? {"Yes" if is_similar else "No"}')

            result = "Yes" if is_similar else "No"

            if result == "Yes":
                item[f"result_{idx}"] = result
                item[f"file_name_{idx}"] = species["file_name"]
                row_results.append(species["file_name"])
                # # Regular expression to match the tree species name
                # species_pattern = r'identified_species\\([^\\]+) -'

                # # Search for the pattern in the file path
                # match = re.search(species_pattern, species["file_name"])

                # Extract and print the tree species name if found
              
                    # species_info = retriever.invoke(f"Scientific name:{tree_species}")
                    
                    # ans = generate_image(species_info, client)
                    # emoji.append(ans)
                    # text_context = [doc.page_content for doc in species_info]
                    # text_context = ", ".join(text_context)
                    # species_context.append(text_context)
                    # print(ans)
                    # species_result.append(tree_species)
    
        img_df.at[img_idx, "species_identified"] = ", ".join(species_result) if species_result else "No similar species found"            
        img_df.at[img_idx, "result_file_path"] = ", ".join(row_results) if row_results else ""
        # img_df.at[img_idx, "emoji"] = ", ".join(emoji) if emoji else ""
        # img_df.at[img_idx, "retreived context"] = ", ".join(species_context) if species_context else ""
        
        
    return cropped_images


# Function to simulate tree species detection

# Display uploaded image and detected tree species
if uploaded_image is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.JPG') as temp_file:
        temp_file.write(uploaded_image.read())
        temp_file_path = temp_file.name
    # Open and display the image
    # image = Image.open(uploaded_image)
    sample_image_path = get_data(temp_file_path)
    boxes = model.predict_image(path=sample_image_path, return_plot=False)
    img_actual = model.predict_image(path=sample_image_path, return_plot=True, color=(137, 0, 0), thickness=9)
    st.image(img_actual, caption='Segmented Panoramic Image', channels ='RGB', use_column_width=True)
    st.success("Sample Dataframe:")
    st.dataframe(boxes.head())
    plt.imshow(img_actual[:,:,::-1])
    # plt.show(img[:,:,::-1])
    plt.savefig(cropped_image_path)
    # if st.button("Next Step"):
  
    accuracy_threshold = st.slider("Accuracy threshold for cropping images:",min_value=0.1, max_value=1.0, value=0.4)
    images_list = split_image_from_dataframe(boxes, temp_file_path, output_folder_name)
    image_width = 200
    st.success("Sample Images:")
    # Display the images in a row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(images_list[3]["image"], caption="Sample 1", width=image_width)

    with col2:
        st.image(images_list[4]["image"], caption="Sample 2", width=image_width)

    with col3:
        st.image(images_list[5]["image"], caption="Sample 3", width=image_width)
    
    folder_path = './identified_species'

    species_feature_list = extract_treespecies_features(folder_path)
    final_result = perform_inference(images_list, species_feature_list, boxes)
    st.success("Final Data:")
    st.dataframe(boxes)
    boxes.to_excel(excel_output_path)
    for index, row in boxes.iterrows():
        species_identified = row['species_identified']
        if species_identified !="No similar species found":
            cropped_image_path = row['cropped_image_path']
            result_file_path = row['result_file_path']
            if type(result_file_path) == list:
                result_file_path = result_file_path[0]
                

            result_file_path = result_file_path.split(',')[0]
            st.write(species_identified)
            col1, col2 = st.columns(2)
            with col1:
                st.image(cropped_image_path, caption='Cropped Image')
            with col2:
                st.image(result_file_path, caption='Species Match')
            
            

            
            

    
    # Detect tree species
    # detected_species = detect_tree_species(image, folder_path)
    
    # Display detected tree species
    # st.write("### Detected Tree Species:")
    # for species in detected_species:
    #     st.write(f"- {species}")



