import time
import torch
from torch.cuda import max_memory_allocated
import torchvision
import argparse
import yaml
from torch.utils.data import DataLoader
from utils import ZeroOneNormalize, CosineAnnealingLRWarmup, evaluate_accuracy_and_loss
from matplotlib import pyplot as plt
import os
from transformers import get_cosine_schedule_with_warmup

os.environ["TORCH_HOME"] = "./pretrained_models"

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./config/classifier_cifar10.yaml", type=str, help="data file path")
args = parser.parse_args()
cfg_path = args.cfg

with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
print(cfg_dict)

visible_device = cfg_dict.get("device")
batchsize = cfg_dict.get("batch_size")
num_workers = cfg_dict.get("num_workers")
num_epoches = cfg_dict.get("epoch")
lr = cfg_dict.get("lr")
weight_decay = cfg_dict.get("weight_decay")
save_dir = cfg_dict.get("save_dir")

train_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(256, 256), antialias=True).cuda(),
    torchvision.transforms.RandomCrop(size=(224, 224)),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(224, 224), antialias=True).cuda(),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)

cifar10_train = torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transforms, download=True)
cifar10_test = torchvision.datasets.CIFAR10(root="./data", train=False, transform=val_transforms, download=True)

train_data_loader = DataLoader(cifar10_train, batch_size=batchsize, drop_last=True, shuffle=True,
                               num_workers=num_workers)
test_data_loader = DataLoader(cifar10_test, batch_size=batchsize, drop_last=False, shuffle=False,
                              num_workers=num_workers)
classes = cifar10_train.classes
print("train: {}, test: {}, classes: {}".format(len(train_data_loader), len(test_data_loader), len(classes)))

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()
model = torch.nn.DataParallel(model, device_ids=visible_device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
loss = torch.nn.CrossEntropyLoss()
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                               num_training_steps=len(train_data_loader) * num_epoches)

train_acc = []
train_loss = []
val_acc = []
val_loss = []
lr_decay_list = []
memory = 0
file_name = os.path.splitext(os.path.basename(__file__))[0]
best_acc = 0.0
best_model = ""
start_time = time.time()
for epoch in range(num_epoches):
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    n = 0
    model.train()

    for batch_idx, (X, y) in enumerate(train_data_loader):
        lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
        # print(lr_decay_list)

        X = X.cuda()
        y = y.cuda()
        y_pred = model(X)
        l = loss(y_pred, y).sum()

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss_sum += l.item()
        train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
        n += y.shape[0]

        if batch_idx % 20 == 0:
            print("epoch: {}, iter: {}, iter loss: {:.4f}, iter acc: {:.4f}".format(epoch, batch_idx, l.item(), (
                    y_pred.argmax(dim=1) == y).float().mean().item()))
        lr_scheduler.step()

    model.eval()
    v_acc, v_loss = evaluate_accuracy_and_loss(test_data_loader, model, loss, accelerator=None)
    train_acc.append(train_acc_sum / n)
    train_loss.append(train_loss_sum / n)
    val_acc.append(v_acc)
    val_loss.append(v_loss)

    if v_acc > best_acc:
        if os.path.exists(os.path.join(save_dir, file_name)) is False:
            os.makedirs(os.path.join(save_dir, file_name))
        best_acc = v_acc
        best_model = os.path.join(os.path.join(save_dir, file_name),
                                  "{}-{}-{}.pth".format(file_name, epoch, best_acc))
        torch.save(model.module.state_dict(), best_model)

    print("epoch: {}, train acc: {:.4f}, train loss: {:.4f}, val acc: {:.4f}, val loss: {:.4f}".format(
        epoch, train_acc[-1], train_loss[-1], val_acc[-1], val_loss[-1]))
    memory = max_memory_allocated()
    print(f'memory allocated: {memory / 1e9:.2f}G')
end_time = time.time()
duration = int(end_time - start_time)
print("duration time: {} s".format(duration))

fig, axes = plt.subplots(1, 3)
axes[0].plot(list(range(1, num_epoches + 1)), train_loss, color="r", label="train loss")
axes[0].plot(list(range(1, num_epoches + 1)), val_loss, color="b", label="validate loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(list(range(1, num_epoches + 1)), train_acc, color="r", label="train acc")
axes[1].plot(list(range(1, num_epoches + 1)), val_acc, color="b", label="validate acc")
axes[1].legend()
axes[1].set_title("Accuracy")

axes[2].plot(list(range(1, len(lr_decay_list) + 1)), lr_decay_list, color="r", label="lr")
axes[2].legend()
axes[2].set_title("Learning Rate")

plt.suptitle('memory: {:.2f} G , duration: {} s'.format(memory / 1e9, duration))
plt.savefig(os.path.join(save_dir, "{}.jpg".format(file_name)))
plt.show()
