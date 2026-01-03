from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a pre-trained YOLOv8 model (YOLOv8n is the nano version; use YOLOv8s, YOLOv8m for larger models)
model = YOLO("yolov8n.yaml")  # Initialize a new YOLOv8 model

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Train the model
model.train(
    data=r"D:\Projects\Boulders And Craters\craters - boulders.v1i.yolov8\data.yaml",  # Path to data.yaml
    epochs=50,                # Number of training epochs
    imgsz=640,                # Image size (default: 640x640)
    batch=16,                 # Batch size
    workers=4,
    device=device                # Number of dataloader workers
    )

# Validate the model on the validation set
metrics = model.val()  # Evaluate performance on validation data
print("Validation Results:", metrics)

# Test the model on the test set
results = model.test(data=r"D:\Projects\Boulders And Craters\craters - boulders.v1i.yolov8\data.yaml")  # Evaluate performance on the test dataset
print("Test Results:", results)

# Predict on new images or datasets
predict_results = model.predict(
    source="test/images",  # Path to test images
    save=True,                    # Save predictions to disk
    conf=0.25                     # Confidence threshold
)
print("Predictions Complete!")

# Export the trained model for deployment
model.export(format="onnx")  # You can export to other formats like TorchScript, CoreML, etc.
