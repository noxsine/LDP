import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



class CLIPProcessor:
    def __init__(self, model_name="ViT-B/16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def process_image(self, image_array):
        image = image_array
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return image

    def process_texts(self, texts):
        return clip.tokenize(texts).to(self.device)

    def get_probabilities(self, image_path, texts):
        image = self.process_image(image_path)
        text = self.process_texts(texts)

        with torch.no_grad():

            logits_per_image, logits_per_text = self.model(image, text)

            probs = logits_per_image.cpu().numpy()

        return probs

clip_processor = CLIPProcessor()
texts = ["A symmetric image at pixel level"]

def truncate_map(x, A=29, B=26.05, y_min=0, y_max=255):
    if x > A:
        return y_min
    elif x < B:
        return y_max
    else:
        return y_min + (y_max - y_min) * (A - x) / (A - B)

def compute_score(image_path_l,image_path_r, patch_size):

    imageL = Image.open(image_path_l).convert('RGB')
    imageR = Image.open(image_path_r).convert('RGB')
    image_arrayL = np.array(imageL)
    image_arrayR = np.array(imageR)


    height, width, _ = image_arrayL.shape


    patch_m = np.zeros((height // patch_size, width // patch_size, 1))

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):

            patchL = image_arrayL[i:i + patch_size, j:j + patch_size, :]
            patchR = image_arrayR[i:i + patch_size, j:j + patch_size, :]
            patchR_flipped = np.fliplr(patchR)
            patch = np.hstack((patchL, patchR_flipped))
            patch = Image.fromarray(patch)

            clip_values = clip_processor.get_probabilities(patch, texts)[0]
            clip_values = truncate_map(clip_values)
            patch_m[i // patch_size, j // patch_size, :] = clip_values

    return patch_m


def save_image(matrix, output_path):

    normalized_matrix = ((matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255).astype(np.uint8)
    #print(normalized_matrix.squeeze(-1).shape)
    normalized_matrix = normalized_matrix.squeeze(-1)
    image = Image.fromarray(normalized_matrix)
    image.save(output_path)



def process_folder(input_folder_L, input_folder_R, output_folder, patch_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_names = [f for f in os.listdir(input_folder_L) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(file_names)

    for idx, file_name in enumerate(file_names):
        print("Image:%d, Remaining:%d" % (idx, total_files-idx))
        image_path_L = os.path.join(input_folder_L, file_name)
        image_path_R = os.path.join(input_folder_R, file_name)
        output_path = os.path.join(output_folder, file_name)

        patch_means_matrix = compute_score(image_path_L, image_path_R, patch_size)
        if patch_means_matrix is not None:
            save_image(patch_means_matrix, output_path)




input_folder_L = 'input/L'  # DP L image
input_folder_R = 'input/R'  # DP R image
output_folder = 'output'  # output
patch_size = 40  # patchsize

process_folder(input_folder_L, input_folder_R, output_folder, patch_size)
