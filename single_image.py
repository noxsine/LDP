import torch
import clip
from PIL import Image
import numpy as np

class CLIPProcessor:
    def __init__(self, model_name="ViT-B/32"):
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
texts = ["A blurry image"]

#
def truncate_map(x, A=26, B=23.3, y_min=0, y_max=255):
    if x > A:
        return y_min
    elif x < B:
        return y_max
    else:
        return y_min + (y_max - y_min) * (A - x) / (A - B)

def compute_clip(image_path, patch_size):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    height, width, _ = image_array.shape

    patch_m = np.zeros((height // patch_size, width // patch_size, 1))

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i:i + patch_size, j:j + patch_size, :]

            patch = Image.fromarray(patch)
            clip_values = clip_processor.get_probabilities(patch, texts)[0]
            clip_values = truncate_map(clip_values)
            patch_m[i // patch_size, j // patch_size, :] = clip_values

    return patch_m


def save_image(matrix, output_path):
    normalized_matrix = ((matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255).astype(np.uint8)

    print(normalized_matrix.squeeze(-1).shape)
    normalized_matrix = normalized_matrix.squeeze(-1)
    image = Image.fromarray(normalized_matrix)

    image.save(output_path)



image_path = 'input/L/1P0A1448.png'  #input
output_path = 'output.png'  #output
patch_size = 40  # patchsize

patch_means_matrix = compute_clip(image_path, patch_size)
save_image(patch_means_matrix, output_path)
