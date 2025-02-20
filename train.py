from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np
import torch
import sys
from minio import Minio


path_images = '/data/images'
path_train_labels = '/data/train_labels'
path_test_labels = '/data/test_labels'
path_save_to = "/data/saved"
numberOfEpochs = 2

# data augmentation
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(50),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    utils.normalize_transform()
])

# %% dataset and dataloader
# trained_labels = ['apple', 'banana']
trained_labels = ['license-plate']
train_dataset = core.Dataset(image_folder=path_images, label_data=path_train_labels, transform=custom_transforms)
test_dataset = core.Dataset(image_folder=path_images, label_data=path_test_labels, transform=custom_transforms)

train_loader = core.DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = core.DataLoader(test_dataset, batch_size=2, shuffle=False)
# %% initialize model
mps_device = torch.device("mps")


model = core.Model(trained_labels)


# %% perform the training
losses = model.fit(train_loader, test_dataset, epochs=numberOfEpochs, verbose=True)

# dave to model.pth
#model.save(path_save_to + '/model.pth')
model.save("/data/model.pth")

# todo, externalize
client = Minio("minio-service:9000",
               access_key="minio",
               secret_key="minio123",
               secure=False
               )
# The file to upload, change this path if needed
source_file = "/data/model.pth"

# The destination bucket and filename on the MinIO server
bucket_name = "models"
destination_file = "model.pth" # really should add a timestamp

found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)
    print("Created bucket", bucket_name)
else:
    print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
client.fput_object( bucket_name, destination_file, source_file,)
print( source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,)