import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

class TransferLearningTrainer:
    def __init__(self, model_type: str, num_classes: int, output_dir="output", learning_rate=1e-3, epochs=10, device=None):
        """
        model_type: 'resnet50' or 'vit'
        num_classes: number of target classes
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_model()

    def _setup_model(self):
        if self.model_type == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.model = resnet50(weights=weights)
            # Replace final layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif self.model_type == "vit":
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError("Unsupported model_type. Choose 'resnet50' or 'vit'.")

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        best_acc = 0.0
        for epoch in range(1, self.epochs+1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Validation after each epoch
            acc = self.evaluate(val_loader, save_results=False)
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"{self.model_type}_best_model.pt"))
                print("Best model saved.")

        # Evaluate final model and save results
        final_acc = self.evaluate(val_loader, save_results=True)
        print(f"Final Validation Accuracy: {final_acc:.4f}")

    def evaluate(self, data_loader: DataLoader, save_results: bool = True):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            "model_type": self.model_type,
            "validation_accuracy": acc,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist()
        }

        if save_results:
            results_path = os.path.join(self.output_dir, "transfer_learning_results.json")
            # Append or create new
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(results)
                with open(results_path, 'w') as f:
                    json.dump(existing, f, indent=4)
            else:
                with open(results_path, 'w') as f:
                    json.dump([results], f, indent=4)
            print(f"Results saved to {results_path}")

        return acc
