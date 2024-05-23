import torch
from torchvision import transforms
from PIL import Image
from model import initialize_model

def load_trained_model(model_name, num_classes, weight_path, mode):
    # 初始化模型
    train_conv = (mode == "all")
    model, _ = initialize_model(model_name, num_classes, train_conv=train_conv, use_pretrained=False)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path, input_size):
    # 定义数据预处理步骤
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 加载并预处理输入图像
    image = Image.open(image_path).convert("RGB")  # 确保图像是RGB格式
    image = data_transforms(image)
    image = image.unsqueeze(0)  # 添加批次维度
    return image

def main(image_path, mode, weight_path):
    # 已确定的参数
    model_name = "resnet50"  # 使用的模型名称
    num_classes = 4  # Fayoum_Banana 数据集中的类别数量
    input_size = 224  # 模型期望的输入尺寸

    # 类别映射字典
    class_names = {0: 'Green', 1: 'Midripen', 2: 'Overripen', 3: 'Yellowish_Green'}

    # 加载训练好的模型
    model = load_trained_model(model_name, num_classes, weight_path, mode)

    # 预处理图像
    image = preprocess_image(image_path, input_size)

    # 输出预处理后的图像
    print("预处理后的图像张量:", image)

    # 进行推断
    with torch.no_grad():
        outputs = model(image)
        print("模型输出:", outputs)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class

# 使用main函数对图像进行分类
if __name__ == "__main__":
    image_path = "FreshApple (1).jpg"  # 输入图像的路径
    mode = "all"  # 设置模式为'all'或'clf'
    weight_path = "path/to/your/trained_model.pth"  # 替换为你训练好的模型权重文件路径
    prediction = main(image_path, mode, weight_path)
    print("预测的类别:", prediction)
