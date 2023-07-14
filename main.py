import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from PIL import Image
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_PATH = './bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

def ReadData():
    # 读取数据
    f1 = open("train.txt", "r")
    lines1 = f1.readlines()
    train_set = []

    for line in lines1[1:]:
        line = line.replace('\n', '')
        guid, tag = line.split(',')
        if tag =='positive':
            label = 0
        elif tag == 'neytral':
            label = 1
        else:
            label = 2
        data = {}
        data['guid'] = guid
        data['label'] = label
        train_set.append(data)

    f2 = open('test_without_label.txt', 'r')
    lines2 = f2.readlines()
    test_set = []
    for line in lines2[1:]:
        data = {}
        data['guid'] = line.split(',')[0]
        test_set.append(data)

    return train_set, test_set


def data_process(dataset):
    for data in dataset:
        guid = data['guid']

        image_path = './data/' + guid + '.jpg'
        # 规范格式，使其符合ResNet的输入
        image = Image.open(image_path).convert('RGB')
        data['image'] = np.array(image.resize((224, 224))).reshape((3, 224, 224))

        text_path = './data/' + guid + '.txt'
        f = open(text_path, 'r', errors = 'ignore')
        text = ''
        for line in f.readlines():
            text += line
        data['text'] = text


def image_classification(train_set, val_set):
    print("----------image_classification----------")
    image_train = []
    image_train_labels = []
    image_valid = []
    image_valid_labels = []

    for data in train_set:
        image_train.append(data['image'])
        image_train_labels.append(data['label'])

    for data in val_set:
        image_valid.append(data['image'])
        image_valid_labels.append(data['label'])

    image_train = torch.from_numpy(np.array(image_train))
    image_train_labels = torch.from_numpy(np.array(image_train_labels))
    image_valid = torch.from_numpy(np.array(image_valid))
    image_valid_labels = torch.from_numpy(np.array(image_valid_labels))

    train_loader = DataLoader(TensorDataset(image_train, image_train_labels), batch_size=100, shuffle=True)
    val_loader = DataLoader(TensorDataset(image_valid, image_valid_labels), batch_size=50)

    image_classifier = ResNet18()
    image_classifier.to(device)

    epoch_num = 20
    learning_rate = 1e-4
    total_step = epoch_num * len(train_loader)

    optimizer = AdamW(image_classifier.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_step, num_training_steps=total_step)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        running_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.shape)
            outputs = image_classifier(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        print('epoch: %d  loss: %.3f' % (epoch+1, running_loss / 35))
        running_loss = 0
    
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, answers = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            answers = answers.to(device)
            outputs = image_classifier(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted.tolist())):
                total_num += answers.size(0)
                correct_num += (predicted == answers).sum().item()

    print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))


def text_classification(train_set, val_set):
    print("----------text_classification----------")
    text_train = []
    text_val = []

    for data in train_set:
        tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
        tokenized_text['label'] = data['label']
        text_train.append(tokenized_text)

    for data in val_set:
        tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
        tokenized_text['label'] = data['label']
        text_val.append(tokenized_text)

    train_loader = DataLoader(TextDataset(text_train), batch_size=25, shuffle=True)
    val_loader = DataLoader(TextDataset(text_val), batch_size=25)

    text_classifier = TextClassifier()
    text_classifier.to(device)

    epoch_num = 20
    learning_rate = 1e-5
    total_step = epoch_num * len(train_loader)

    optimizer = AdamW(text_classifier.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_step, num_training_steps=total_step)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        running_loss = 0
        for i, data in enumerate(train_loader):
            input_ids, attn_mask, labels = data
            input_ids = torch.tensor([item.numpy() for item in input_ids])
            attn_mask = torch.tensor([item.numpy() for item in attn_mask])
            input_ids = input_ids.T
            attn_mask = attn_mask.T
            # labels = torch.tensor([item.numpy() for item in labels])
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            outputs = text_classifier(input_ids, attn_mask)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            running_loss += loss.item()
        print('epoch: %d  loss: %.3f' % (epoch+1, running_loss/140))
        running_loss = 0

    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for data in val_loader:
            input_ids, attn_mask, labels = data
            input_ids = torch.tensor([item.numpy() for item in input_ids])
            input_ids = input_ids.T
            attn_mask = torch.tensor([item.numpy() for item in attn_mask])
            attn_mask = attn_mask.T
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
        
            outputs = text_classifier(input_ids, attn_mask)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted.tolist())):
                total_num += labels.size(0)
                correct_num += (predicted == labels).sum().item()

        print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))

def dataset_process(dataset):
    for data in dataset:
        tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
        data['input_ids'] = tokenized_text['input_ids']
        data['attn_mask'] = tokenized_text['attention_mask']

def multimodel(train_set, val_set, test_set):
    print("----------multimodel----------")
    dataset_process(train_set)
    dataset_process(val_set)
    dataset_process(test_set)

    train_loader = DataLoader(MultimodalDataset(train_set), batch_size=25, shuffle=True)
    valid_loader = DataLoader(MultimodalDataset(val_set), batch_size=25)
    test_loader = DataLoader(MultimodalDataset(test_set), batch_size=25)

    image_classifier = ResNet18()
    image_classifier.to(device)
    text_classifier = TextClassifier()
    text_classifier.to(device)

    multimodal_model = MultimodalModel(image_classifier=image_classifier, text_classifier=text_classifier, output_features=100, image_weight=0.5, text_weight=0.5)
    multimodal_model.to(device)

    epoch_num = 10
    learning_rate = 1e-5
    total_step = epoch_num * len(train_loader)

    optimizer = AdamW(multimodal_model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_step, num_training_steps=total_step)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        running_loss = 0
        for i, data in enumerate(train_loader):
            _, input_ids, attn_mask, image, label = data
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            image = image.to(device)
            image = image.float()
            label = label.to(device)

            outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
            # print(outputs.shape)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        print('epoch: %d  loss: %.3f' % (epoch+1, running_loss/140))
        running_loss = 0

    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for data in valid_loader:
            _, input_ids, attn_mask, image, label = data
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            image = image.to(device)
            image = image.float()
            label = label.to(device)
        
            outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted.tolist())):
                total_num += label.size(0)
                correct_num += (predicted == label).sum().item()

    print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))

    test_dict = {}
    with torch.no_grad():
        for data in test_loader:
            guid, input_ids, attn_mask, image, label = data
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            image = image.to(device)
            image = image.float()
            label = label.to(device)
        
            outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.tolist()
            for i in range(len(predicted)):
                id = guid[i]
                test_dict[id] = predicted[i]

    with open('./test_without_label.txt', 'r') as f:
        lines = f.readlines()

    f1 = open('./test.txt', 'w')
    f1.write(lines[0])

    for line in lines[1:9]:
        # print(line)
        guid = line.split(',')[0]
        f1.write(guid)
        f1.write(',')
        label = test_dict[guid]
        if label == 0:
            f1.write('positive\n')
        elif label == 1:
            f1.write('neutral\n')
        else:
            f1.write('negative\n')

if __name__ == '__main__':
    train_set, test_set = ReadData()
    data_process(train_set) # guid, label, image, text的形式
    data_process(test_set)

    # 划分数据集，设置为0.2
    train_num = 3200
    val_num = 800
    train_train_set, train_val_set = random_split(train_set, [train_num, val_num])

    # image_classification(train_train_set, train_val_set)
    # text_classification(train_train_set, train_val_set)
    multimodel(train_train_set, train_val_set, test_set)