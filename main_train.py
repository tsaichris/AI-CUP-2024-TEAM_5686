import torch
import matplotlib.pyplot as plt
import os
import cv2
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from UNet_modified import UNet, DICE_BCE_Loss, dice_coeff
from edge import edge_detection
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"

# Data augmentation
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(428, 428),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.3),  
            A.Blur(blur_limit=3, always_apply=False, p=0.1),  
            A.MotionBlur(blur_limit=3, always_apply=False, p=0.1), 
            A.MedianBlur(blur_limit=3, always_apply=False, p=0.1), 
            A.GaussianBlur(blur_limit=3, always_apply=False, p=0.1),  
            #A.CLAHE(clip_limit=1.0, tile_grid_size=(4, 4), always_apply=False, p=0.1), 
            A.RandomGamma(gamma_limit=(90, 110), always_apply=False, p=0.1),  
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, always_apply=False, p=0.1),  
            A.GaussNoise(var_limit=(5.0, 20.0), always_apply=False, p=0.1),  
            A.Transpose(always_apply=False, p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5)),  # Normalize both RGB and edge
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(428, 428),
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5)),  
            ToTensorV2()
        ])


# Data pre-process
class CustomDataset(Dataset):
    def __init__(self, image_list, image_folder, mask_folder, transform=None):
        self.image_list = image_list
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_folder, img_name)
        mask_path = os.path.join(self.mask_folder, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize the mask
        mask = mask / 255.0

        edge_mask, _ = edge_detection(image)
        
        edge_mask = edge_mask[:, :, np.newaxis]  # Add channel dimension to edge mask
        combined_image = np.concatenate((image, edge_mask), axis=2)  # Combine image and edge mask

        if self.transform:
            augmented = self.transform(image=combined_image, mask=mask)
            #augmented = self.transform(image=image, mask=mask)
            combined_image = augmented['image']
            #image = augmented['image']
            mask = augmented['mask']

        # Convert mask to float and add a channel dimension
        mask = mask.float().unsqueeze(0)

        return combined_image, mask
        #return image, mask

def train(model, trainloader, valloader, optimizer, loss_fn, scheduler, epochs=10):
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    best_val_dice = 0.0  # Initialize Dice coefficient

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        # train
        for images, masks in trainloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_coeff(logits, masks)
        train_loss /= len(trainloader)
        train_dice /= len(trainloader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in valloader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()
                val_dice += dice_coeff(logits, masks)
        val_loss /= len(valloader)
        val_dice /= len(valloader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"Epoch: {epoch + 1}  Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Save the model if the validation Dice coefficient is the best 
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation Dice coefficient: {best_val_dice:.4f}")

    return train_losses, train_dices, val_losses, val_dices

def main():
    # Paths
    base_path = 'Training_dataset'
    image_folder = os.path.join(base_path, 'img')
    mask_folder = os.path.join(base_path, 'label_img_converted')
    print(base_path)

    # Load images and split into train, test, validation sets
    images = os.listdir(image_folder)
    train_images, test_images = train_test_split(images, test_size=0.01, random_state=42)
    test_images, val_images = train_test_split(test_images, test_size=0.5, random_state=42)

    # Dataset sizes
    dataset_sizes = [len(train_images), len(test_images), len(val_images)]
    labels = ["Train", "Test", "Val"]

    plt.bar(labels, dataset_sizes)
    plt.show()

    # Transforms
    train_transform = get_transforms(train=True)
    val_test_transform = get_transforms(train=False)

    # Datasets
    trainset = CustomDataset(train_images, image_folder, mask_folder, transform=train_transform)
    testset = CustomDataset(test_images, image_folder, mask_folder, transform=val_test_transform)
    valset = CustomDataset(val_images, image_folder, mask_folder, transform=val_test_transform)

    # DataLoaders
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 8

    trainloader = DataLoader(trainset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    valloader = DataLoader(valset, batch_size=val_batch_size, num_workers=2, shuffle=True)
    #testloader = DataLoader(testset, batch_size=test_batch_size, num_workers=2, shuffle=True)

    epochs = 50
    loss_fn = DICE_BCE_Loss()
    
    model = UNet(4, 1).to(device)
    state_dict = torch.load('428.pth')
    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Initialize the dynamic learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # train the model
    train_losses, train_dices, val_losses, val_dices = train(model, trainloader, valloader, optimizer, loss_fn, scheduler, epochs)
    
    # Visualizet the training result
    plt.figure(figsize= (10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), train_dices)
    plt.plot(np.arange(epochs), val_dices)
    plt.xlabel("Epoch")
    plt.ylabel("DICE Coeff")
    plt.legend(["Train DICE", "Val DICE"])
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), train_losses)
    plt.plot(np.arange(epochs), val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train Loss", "Val Loss"])
    plt.savefig('train_result6')

   
if __name__ == '__main__':
    main()