import torch
from torchvision import transforms
from PIL import Image
from model import initialize_model

def main(image_path):
    # 已确定的参数
    model_name = "resnet50"  # 使用的模型名称
    num_classes = 10  # Fayoum数据集中的类别数量
    input_size = 224  # 模型期望的输入尺寸
    seed = 42  # 随机种子
    n_train_samples = 40  # 训练样本数量

    # 设置随机种子
    torch.manual_seed(seed)

    # 初始化模型
    model, _ = initialize_model(model_name, num_classes, train_conv=False, use_pretrained=True)
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
        return predicted.item()

# 使用main函数对图像进行分类
if __name__ == "__main__":
    image_path = "FreshApple (1).jpg"  # 输入图像的路径
    prediction = main(image_path)
    print("预测的类别索引:", prediction)
