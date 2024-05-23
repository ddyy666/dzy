import torch
from torchvision import transforms
from PIL import Image
from model import initialize_model

def main(image_path, mode):
    # 已确定的参数
    model_name = "resnet50"  # 使用的模型名称
    num_classes = 4  # Fayoum_Banana 数据集中的类别数量
    input_size = 224  # 模型期望的输入尺寸
    seed = 42  # 随机种子
    n_train_samples = 40  # 训练样本数量

    # 类别映射字典
    class_names = {0: 'Green', 1: 'Midripen', 2: 'Overripen', 3: 'Yellowish_Green'} 

    # 设置随机种子
    torch.manual_seed(seed)

    # 初始化模型
    train_conv = False
    if mode == "all":
        train_conv = True
    model, _ = initialize_model(model_name, num_classes, train_conv=train_conv, use_pretrained=True)
    model.eval()

    # 定义数据预处理步骤
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载并预处理输入图像
    image = Image.open(image_path)
    image = data_transforms(image)
    image = image.unsqueeze(0)

    # 进行推断
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class

# 使用main函数对图像进行分类
if __name__ == "__main__":
    image_path = "FreshApple (1).jpg"  # 输入图像的路径
    mode = "all"  # 设置模式为'all'或'clf'
    prediction = main(image_path, mode)
    print("预测的类别:", prediction)
