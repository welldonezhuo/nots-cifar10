import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def unnormalize(img):
    """Undo normalization for display."""
    for t, m, s in zip(img, MEAN, STD):
        t.mul_(s).add_(m)
    return img.clamp_(0, 1)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def log_predictions(writer, model, images, labels, device, epoch):
    """Log fixed test images side by side with horizontal bar charts of class probabilities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_images = len(labels)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu()
    preds = probs.argmax(dim=1)

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 3 * num_images),
                             gridspec_kw={"width_ratios": [1, 2]})
    for i in range(num_images):
        img = unnormalize(images[i].cpu().clone()).permute(1, 2, 0).numpy()
        correct = preds[i] == labels[i]

        # Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {CLASSES[labels[i]]}", fontsize=11)
        axes[i, 0].axis("off")

        # Bar chart
        colors = ["green" if c == labels[i] else "salmon" for c in range(10)]
        colors[preds[i]] = "green" if correct else "red"
        axes[i, 1].barh(range(10), probs[i].numpy(), color=colors)
        axes[i, 1].set_yticks(range(10))
        axes[i, 1].set_yticklabels(CLASSES, fontsize=9)
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_title(f"Pred: {CLASSES[preds[i]]} ({'✓' if correct else '✗'})", fontsize=11)
        axes[i, 1].invert_yaxis()

    fig.tight_layout()
    writer.add_figure("Predictions", fig, epoch)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Fix 5 images for consistent prediction visualization across epochs
    viz_images = torch.stack([testset[i][0] for i in range(5)]).to(device)
    viz_labels = torch.tensor([testset[i][1] for i in range(5)])

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(trainloader)

        # Test
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        test_loss = test_loss / len(testloader)
        scheduler.step()

        # Log scalars — grouped so train/test appear on the same chart
        writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Log sample predictions every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            log_predictions(writer, model, viz_images, viz_labels, device, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    writer.close()


if __name__ == "__main__":
    main()
