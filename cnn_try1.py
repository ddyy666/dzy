from PIL import Image
import torchvision.transforms as transforms
from model import initialize_model

# 初始化模型
model_name = "resnet50"  # 选择模型名称
num_classes = 10  # 类别数量
model, _ = initialize_model(model_name, num_classes)

# 加载并转换图像
image_path = "image.jpg"  # 图像文件路径
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])
input_image = transform(image).unsqueeze(0)  # 添加批量维度

# 使用模型进行推理
model.eval()
with torch.no_grad():
    output = model(input_image)

# 处理输出结果
# 这里您可以根据模型的输出格式进行后续处理，例如应用softmax获取概率分布、解码类别标签等
