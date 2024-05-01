from Model import NeuralNetwork, read_images, read_labels
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

# 加载测试集数据
test_images = read_images('t10k-images-idx3-ubyte.gz')
test_labels = read_labels('t10k-labels-idx1-ubyte.gz')

# 从文件中加载 best_params
with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

best_hidden_size = best_params['hidden_size']
# 初始化神经网络模型
model = NeuralNetwork(input_size=test_images.shape[1], hidden_size1=best_hidden_size, hidden_size2=best_hidden_size, output_size=len(np.unique(test_labels)), activation='relu')

# 加载训练好的模型权重
best_model_weights = np.load('best_model_weights.npy', allow_pickle=True)
model.W1, model.b1, model.W2, model.b2, model.W3, model.b3 = best_model_weights

# 使用模型进行预测
probabilities = model.forward(test_images)
predictions = np.argmax(probabilities, axis=1)

# 计算分类准确率
accuracy = accuracy_score(test_labels, predictions)
print('测试集准确率:', accuracy)


