# AI-CUP-2024-TEAM_5686

## Step 1: Clone the Repository

git clone <https://github.com/tsaichris/AI-CUP-2024-TEAM_5686.git>

## Step 2: Set Up the Necessary Environment
conda create --name test python=3.8

conda activate test

pip install -r environment.yml

## Step 3: Convert Ground Truth Dataset from .PNG to .JPG

Run JPG2PNG.py to convert all ground truth dataset (label_img) from .png format to .jpg format.

Note: Remember to change the path name in the script.

## Step 4: Train the Model
Run main_train.py to train the model.

Note: Remember to change the path name in the script.

## Step 5: Download Trained Weights and Generate Test Results
link:https://drive.google.com/file/d/1khDYeYbQqhQoUOO4i32kvuL6tSyzgrrp/view?usp=sharing

Run test_4c.py to generate the test results.

Note: Remember to change the path name in the script.

## Test environment in colab:
link:https://colab.research.google.com/drive/1j4ipN-O4W-QLbTjdck-IwzJvk436mFTP?usp=sharing

Note: package list can be referred from "environment_colab.yaml"
