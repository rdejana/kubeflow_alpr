#%% packages
from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np
import torch


labels = ['license-plate']

model = core.Model.load('saved/model.pth', labels)

test_image_path = 'images/80g2l.jpg'
test_image = utils.read_image(test_image_path)
pred = model.predict(test_image)

labels, boxes, scores = pred
conf_threshold = 0.70
filtered_indices = np.where(scores > conf_threshold)
filtered_scores = scores[filtered_indices]
filtered_boxes = boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
print(filtered_scores)
show_labeled_image(test_image, filtered_boxes, filtered_labels)