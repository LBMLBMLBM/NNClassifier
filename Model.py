import numpy as np
import gzip
import matplotlib.pyplot as plt

# 定义激活函数及其导数
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.activation = activation

        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * 0.01
        self.b1 = np.zeros((1, self.hidden_size1))
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.01
        self.b2 = np.zeros((1, self.hidden_size2))
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.01
        self.b3 = np.zeros((1, self.output_size))

    def forward(self, x):
        # 前向传播
        hidden_input1 = np.dot(x, self.W1) + self.b1
        if self.activation == 'sigmoid':
            hidden_output1 = sigmoid(hidden_input1)
        elif self.activation == 'relu':
            hidden_output1 = relu(hidden_input1)
        else:
            raise ValueError("Unsupported activation function")

        hidden_input2 = np.dot(hidden_output1, self.W2) + self.b2
        if self.activation == 'sigmoid':
            hidden_output2 = sigmoid(hidden_input2)
        elif self.activation == 'relu':
            hidden_output2 = relu(hidden_input2)
        else:
            raise ValueError("Unsupported activation function")

        output = np.dot(hidden_output2, self.W3) + self.b3
        probabilities = softmax(output)
        return probabilities

    def backward(self, x, y, learning_rate, reg_lambda):
        d_output = self.probabilities - y # 输出层梯度

        d_output /= x.shape[0] # 输出层的平均梯度

        d_weights3 = np.dot(self.hidden_output2.T, d_output) # 输出层权重的梯度
        d_bias3 = np.sum(d_output, axis=0, keepdims=True) # 输出层每个神经元的偏置的梯度

        if self.activation == 'sigmoid':
            d_hidden2 = np.dot(d_output, self.W3.T) * sigmoid_derivative(self.hidden_output2)
        elif self.activation == 'relu':
            d_hidden2 = np.dot(d_output, self.W3.T) * relu_derivative(self.hidden_output2)
        else:
            raise ValueError("Unsupported activation function")

        d_weights2 = np.dot(self.hidden_output1.T, d_hidden2)
        d_bias2 = np.sum(d_hidden2, axis=0, keepdims=True)

        if self.activation == 'sigmoid':
            d_hidden1 = np.dot(d_hidden2, self.W2.T) * sigmoid_derivative(self.hidden_output1)
        elif self.activation == 'relu':
            d_hidden1 = np.dot(d_hidden2, self.W2.T) * relu_derivative(self.hidden_output1)
        else:
            raise ValueError("Unsupported activation function")

        d_weights1 = np.dot(x.T, d_hidden1)
        d_bias1 = np.sum(d_hidden1, axis=0, keepdims=True)

        # 加上L2正则化项的梯度
        d_weights3 += reg_lambda * self.W3
        d_weights2 += reg_lambda * self.W2
        d_weights1 += reg_lambda * self.W1

        # 更新权重和偏置
        self.W1 -= learning_rate * d_weights1
        self.b1 -= learning_rate * d_bias1
        self.W2 -= learning_rate * d_weights2
        self.b2 -= learning_rate * d_bias2
        self.W3 -= learning_rate * d_weights3
        self.b3 -= learning_rate * d_bias3

class NeuralNetworkTrainer:
    def __init__(self, model):
        self.model = model
        self.best_val_acc = 0
        self.best_model = None
        self.loss_history = []
        self.acc_history = []

    def train(self, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=64, learning_rate=0.01, reg_lambda=0.001, verbose=True):
        num_batches = X_train.shape[0] // batch_size

        for epoch in range(num_epochs):
            # Shuffle训练数据
            shuffle_idx = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

            # 分批训练
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                # 前向传播
                probabilities = self.model.forward(X_batch)

                # 将标签转换为one-hot编码
                y_one_hot = np.eye(self.model.output_size)[y_batch]

                # 计算损失
                data_loss = -np.sum(np.log(probabilities) * y_one_hot) / len(y_batch)

                # 反向传播
                self.model.probabilities = probabilities
                self.model.hidden_output1 = relu(np.dot(X_batch, self.model.W1) + self.model.b1)
                self.model.hidden_output2 = relu(np.dot(self.model.hidden_output1, self.model.W2) + self.model.b2)
                self.model.backward(X_batch, y_one_hot, learning_rate, reg_lambda)

            # 在验证集上评估
            val_probabilities = self.model.forward(X_val)
            val_predictions = np.argmax(val_probabilities, axis=1)
            val_acc = accuracy_score(y_val, val_predictions)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model = (
                    self.model.W1.copy(),
                    self.model.b1.copy(),
                    self.model.W2.copy(),
                    self.model.b2.copy(),
                    self.model.W3.copy(),
                    self.model.b3.copy()
                )

            # 收集损失和准确率以便绘图
            self.loss_history.append(data_loss)
            self.acc_history.append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}")

        # 保存最佳模型参数
        self.model.W1, self.model.b1, self.model.W2, self.model.b2, self.model.W3, self.model.b3 = self.best_model
        np.save('best_model_weights.npy', self.best_model)

    def plot_history(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.acc_history, label='Accuracy', color='orange')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

# 定义读数据的方法
def read_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 读取前16个字节
        magic = f.read(4)
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # 读取所有图像数据并转换为NumPy数组
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols) / 255.0
    return images


def read_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 读取前8个字节
        magic = f.read(4)
        num_items = int.from_bytes(f.read(4), 'big')
        # 读取所有标签数据并转换为NumPy数组
        labels = np.frombuffer(f.read(),dtype=np.uint8)
    return labels

