# coding:utf-8
import os
import torch
import torchvision
from imutils.video import fps
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch import nn
from utils import ZeroOneNormalize
from torch.utils.data import DataLoader

os.environ["TORCH_HOME"] = "./pretrained_models"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def predict(model_path, classes, device=None, is_half=None, is_amp=None):
    print("#" * 40, model_path, "#" * 40)

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()
    # print(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.to(device)
    if is_half:
        model = model.half()
    model.eval()

    fps_speed = fps.FPS()
    fps_speed.start()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_data_loader):
            X = X.cuda()
            y = y.cuda()

            # 半精度
            if is_half:
                X = X.half()

            # 混合精度
            if is_amp:
                with torch.cuda.amp.autocast():
                    res = model(X)
            else:
                res = model(X)

            cls_index = res.argmax(dim=1)
            cls_prob = nn.functional.softmax(res, dim=1)

            pred_prob = cls_prob[0][cls_index].item()
            pred_cls = classes[cls_index]

            print("true label: {}, pred label: {}, probability: {}".format(classes[y.item()], pred_cls, pred_prob))

            fps_speed.update()

            if batch_idx > 10:
                break
    fps_speed.stop()
    print("FPS: {}".format(fps_speed.fps()))
    print("time: {}".format(fps_speed.elapsed()))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transforms_list = [
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.Resize(size=(224, 224), antialias=True).cuda(),
        ZeroOneNormalize(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    val_transforms = torchvision.transforms.Compose(val_transforms_list)

    cifar10_test = torchvision.datasets.CIFAR10(root="./data", train=False, transform=val_transforms, download=True)
    test_data_loader = DataLoader(cifar10_test, batch_size=1, drop_last=False, shuffle=False, num_workers=8)
    classes = cifar10_test.classes

    model_args = [
        {"model_path": "results/pytorch_SingleGPU/pytorch_SingleGPU-4-0.8502.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/pytorch_half_precision/pytorch_half_precision-3-0.7657.pth", "is_half": True, "is_amp": None},
        {"model_path": "results/pytorch_auto_mixed_precision/pytorch_auto_mixed_precision-4-0.8527.pth", "is_half": None, "is_amp": True},
        {"model_path": "results/pytorch_DP/pytorch_DP-4-0.853.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/pytorch_DDP/pytorch_DDP-4-0.8498.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/pytorch_torchrun_DDP/pytorch_torchrun_DDP-4-0.8535.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/accelerate_DDP/accelerate_DDP-4-0.8429588607594937.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/pytorch_FSDP/pytorch_FSDP-4-0.854.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/pytorch_torchrun_FSDP/pytorch_torchrun_FSDP-3-0.8445.pth", "is_half": None, "is_amp": None},
        {"model_path": "results/accelerate_FSDP/accelerate_FSDP-4-0.8551226265822784.pth", "is_half": None, "is_amp": None},
    ]
    for model_info in model_args:
        predict(**model_info, classes=classes, device=device)
