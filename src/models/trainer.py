import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

class SatelliteCNNTrainer:
    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=.0001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        # track the training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        # function to train 1 epoch
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc='Training') # progress bar

        for batch_index, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # record stats
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # progress bar update
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_index+1):.4f}',
                'Accuracy': f'{accuracy:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc
    
    def validate(self):
        # eval the model
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = val_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc
    
    def train(self, epochs=10, save_best=True):
        # training loop
        print(f'Training on device: {self.device}\nTraining samples: {len(self.train_loader.dataset)}\nValidation samples: {len(self.val_loader.dataset)}')
        print('-'*22)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # print epoch results
            print(f'\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%\nVal Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_satellite_terrain_model.pth')
                print(f'\nNew best model saved with Validation Accuracy: {val_acc:.2f}%!')
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.0f}s')
        print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

        return {'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accs': self.train_accs,
                'val_accs': self.val_accs,
                'best_val_acc': best_val_acc
                }