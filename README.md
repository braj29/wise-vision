# wise-vision
Repository for project done at Gainforest Hackathon ETH AI  Center XPRIZE Competition

title: Wise Vision App
emoji: 🌍
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
This project is result of submission to the ETH AI Center GainForest Hackathon for XPrize.

The project repository contains Deepforest and DETECT-TREE2 models, which are image segmentation models.

Initially, the App asks user to upload an .jpb Panoramic image, it is automatically segmented and the output is a excel file with co-ordinates.

The segmented trees from the panoramic images are compared with identified species and results are generated on this.

The identification/classification is done using a VGG-16 model which is pretrained on 1 million images.

Testing of this model is done by a Dataset created for this project: https://segments.ai/r/drone-panoramic/.

This dataset has species annotated in the panoramic images. The results of the classification from the VGG or any other model can be tested with this dataset.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference