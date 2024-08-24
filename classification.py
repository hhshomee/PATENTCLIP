import pandas as pd
import os
import torch
from PIL import Image
import open_clip
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:patentclip/PatentCLIP_Vit_B', device=device)
tokenizer = open_clip.get_tokenizer('hf-hub:patentclip/PatentCLIP_Vit_B')
df_train=pd.read_csv('train_2023.csv')
df_val=pd.read_csv('val_2023.csv')
category_to_idx = {category: idx for idx, category in enumerate(df_train['cat'].unique())}
class PatentDataset(Dataset):
    def __init__(self, df,tokenizer,category_to_idx):
        self.images = df['full_path'].tolist()
        self.category = df['cat'].tolist()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.tokenize = tokenizer
        self.category_to_idx = category_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transform(Image.open(str(self.images[idx])))
        categories = self.tokenize([str(self.category[idx])])[0]
        category = self.category_to_idx[self.category[idx]]  
        return images, category
    
train_loader = DataLoader(PatentDataset(df_train, tokenizer,category_to_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(PatentDataset(df_val, tokenizer,category_to_idx), batch_size=32, shuffle=False)
import torch.nn as nn


class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  
        return self.classifier(features)
num_class = df_train['cat'].nunique()
print(num_class)
model_ft = CLIPFineTuner(model, num_class).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

num_epochs = 5

for epoch in range(num_epochs):
    model_ft.train()  
    running_loss = 0.0  
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000") 

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)  
        optimizer.zero_grad()  
        outputs = model_ft(images)  
        loss = criterion(outputs, labels) 
        loss.backward()  
        optimizer.step() 

        running_loss += loss.item()  
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}") 

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}') 
    
    model_ft.eval()  
    correct = 0  
    total = 0  

    with torch.no_grad():  
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model_ft(images) 
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  

    print(f'Validation Accuracy: {100 * correct / total}%')  


torch.save(model_ft.state_dict(), 'clip_finetuned_our.pth')  
