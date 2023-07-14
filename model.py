import torch

import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from torch.utils.data import Dataset
import torchvision.models as models


# 采用lab3-图像分类及经典CNN实现中的ResNet模型
class ResBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
    self.bn1 = nn.BatchNorm2d(output_channel)
    self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
    self.bn2 = nn.BatchNorm2d(output_channel)
  
  def forward(self, x):
    output = self.conv1(x)
    output = self.bn1(output)
    output = F.relu(output)
    output = self.conv2(x)
    output = self.bn2(output)
    output = F.relu(output + x)
    return output


class ShortcutResBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(ShortcutResBlock, self).__init__()
    self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=2)
    self.bn1 = nn.BatchNorm2d(output_channel)
    self.conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1, stride=2)
    self.bn2 = nn.BatchNorm2d(output_channel)
    self.conv3 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
    self.bn3 = nn.BatchNorm2d(output_channel)

  def forward(self, x):
    output1 = self.conv1(x)
    output1 = self.bn1(output1)
    output2 = self.conv2(x)
    output2 = self.bn2(output2)
    output2 = F.relu(output2)
    output2 = self.conv3(output2)
    output2 = self.bn3(output2)
    output = F.relu(output1 + output2)
    return output

class ResNet18(nn.Module):
  def __init__(self):
    super(ResNet18, self).__init__()
    self.model = models.resnet18(pretrained=False)
    self.model.load_state_dict(torch.load('./resnet18-f37072fd.pth'))
    # 冻结特征提取层
    for param in self.model.parameters():
      param.requires_grad = False
    # 获取ResNet-18模型的全连接层输入特征数
    num_features = self.model.fc.in_features
    # 替换全连接层为输出数目为3的新层
    self.model.fc = torch.nn.Linear(num_features, 3)
  def forward(self, x):
    return self.model(x)
  

class TextClassifier(nn.Module):
  def __init__(self):
    super(TextClassifier, self).__init__()
    BERT_PATH = './bert-base-chinese'
    self.model = AutoModel.from_pretrained(BERT_PATH)
    self.model = self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    self.dropout = nn.Dropout(0)
    # self.model.to(device)
    self.fc = nn.Linear(768, 3)
  
  def forward(self, x, attn_mask=None):
    x = x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    attn_mask = attn_mask.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    output = self.model(x, attention_mask=attn_mask)
    # output = output.to(device)
    output = output[1]
    output = torch.flatten(output, 1)
    output = self.fc(output)
    return output
  
class TextDataset(Dataset):
  def __init__(self, data):
    super(TextDataset, self).__init__()
    self.data = data
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    input_ids = self.data[idx]['input_ids']
    attn_mask = self.data[idx]['attention_mask']
    label = self.data[idx]['label']
    return input_ids, attn_mask, label
  

class MultimodalDataset(Dataset):
  def __init__(self, data):
    super(MultimodalDataset, self).__init__()
    self.data = data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    guid = self.data[idx]['guid']
    input_ids = torch.tensor(self.data[idx]['input_ids'])
    attn_mask = torch.tensor(self.data[idx]['attn_mask'])
    image = torch.tensor(self.data[idx]['image'])
    label = self.data[idx].get('label')
    if label is None:
      label = -100
    label = torch.tensor(label)
    return guid, input_ids, attn_mask, image, label
  

class MultimodalModel(nn.Module):
  def __init__(self, image_classifier, text_classifier, output_features, image_weight=0.5, text_weight=0.5):
    super(MultimodalModel, self).__init__()
    self.image_classifier = image_classifier
    self.text_classifier = text_classifier
    # 将最后的全连接层删除
    self.image_classifier.model.fc = nn.Sequential()  # (batch_num, 512)
    self.text_classifier.fc = nn.Sequential()    # (batch_num, 768)
    # 文本特征向量和图片特征向量的权重, 默认均为0.5
    self.image_weight = image_weight
    self.text_weight = text_weight
    self.fc1 = nn.Linear((512+768), output_features)
    self.fc2 = nn.Linear(output_features, 3)

  def forward(self, input_ids, attn_mask, image):
    image_output = self.image_classifier(image)
    text_output = self.text_classifier(input_ids, attn_mask)
    output = torch.cat([image_output, text_output], dim=-1)
    output = self.fc1(output)
    output = self.fc2(output)
    return output