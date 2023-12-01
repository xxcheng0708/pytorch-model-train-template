# coding:utf-8
import os

import torch
import onnx
import onnxruntime
import numpy as np
import time
import torchvision

os.environ["TORCH_HOME"] = "./pretrained_models"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def pytorch_2_onnx(model_path, export_model_path):
    """
    将pytorch模型导出为onnx，导出时pytorch内部使用的是trace或者script先执行一次模型推理，然后记录下网络图结构
    所以，要导出的模型要能够被trace或者script进行转换
    :return:
    """
    # 加载预训练模型
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()
    # print(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # pytorch转换为onnx内部使用trace或者script，需要提供一组输入数据执行一次模型推理过程，然后进行trace记录
    dummy_input = torch.randn(4, 3, 224, 224, device="cuda")
    input_names = ["input_data"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output_data"]

    torch.onnx.export(
        model,  # pytorch网络模型
        dummy_input,  # 随机的模拟输入
        export_model_path,  # 导出的onnx文件位置
        export_params=True,  # 导出训练好的模型参数
        verbose=10,  # debug message
        training=torch.onnx.TrainingMode.EVAL,  # 导出模型调整到推理状态，将dropout，BatchNorm等涉及的超参数固定
        input_names=input_names,  # 为静态网络图中的输入节点设置别名，在进行onnx推理时，将input_names字段与输入数据绑定
        output_names=output_names,  # 为输出节点设置别名
        # 如果不设置dynamic_axes，那么对于输入形状为[4, 3, 224, 224]，在以后使用onnx进行推理时也必须输入[4, 3, 224, 224]
        # 下面设置了输入的第0维是动态的，以后推理时batch_size的大小可以是其他动态值
        dynamic_axes={
            # a dictionary to specify dynamic axes of input/output
            # each key must also be provided in input_names or output_names
            "input_data": {0: "batch_size"},
            "output_data": {0: "batch_size"}
        })
    return export_model_path


def onnx_check(model_path):
    """
    验证导出的模型格式时候正确
    :param model_path:
    :return:
    """
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))


def onnx_inference(model_path):
    """
    模型推理
    :param model_path:
    :return:
    """
    # 使用onnxruntime-gpu在GPU上进行推理
    session = onnxruntime.InferenceSession(model_path,
                                           providers=[
                                               ("CUDAExecutionProvider", {  # 使用GPU推理
                                                   "device_id": 0,
                                                   "arena_extend_strategy": "kNextPowerOfTwo",
                                                   "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                                                   "cudnn_conv_algo_search": "EXHAUSTIVE",
                                                   "do_copy_in_default_stream": True,
                                                   # "cudnn_conv_use_max_workspace": "1"    # 在初始化阶段需要占用好几G的显存
                                               }),
                                               "CPUExecutionProvider"  # 使用CPU推理
                                           ])
    # session = onnxruntime.InferenceSession(model_path)
    data = np.random.randn(2, 3, 224, 224).astype(np.float32)

    # 获取模型原始输入的字段名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("input name: {}".format(input_name))

    # 以字典方式将数据输入到模型中
    outputs = session.run([output_name], {input_name: data})
    print(outputs)


if __name__ == '__main__':
    model_path = pytorch_2_onnx("results/pytorch_SingleGPU/pytorch_SingleGPU-4-0.8502.pth", "results/pytorch_SingleGPU/pytorch_SingleGPU.onnx")

    onnx_check(model_path)

    onnx_inference(model_path)
