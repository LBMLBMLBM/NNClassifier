from Model import *

def parameter_search(X_train, y_train, X_val, y_val, learning_rates, hidden_layer_sizes, reg_strengths, num_epochs=20, batch_size=64):
    best_acc = 0
    best_params = {}
    best_acc_per_params = {}

    for lr in learning_rates:
        for hidden_size in hidden_layer_sizes:
            for reg_strength in reg_strengths:
                # 创建神经网络模型
                model = NeuralNetwork(input_size=X_train.shape[1], hidden_size1=hidden_size, hidden_size2=hidden_size, output_size=len(np.unique(y_train)), activation = 'relu')
                trainer = NeuralNetworkTrainer(model)

                # 训练模型
                trainer.train(X_train, y_train, X_val, y_val, num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, reg_lambda=reg_strength, verbose=False)

                # 获取最佳验证准确率
                acc = trainer.best_val_acc

                # 更新最佳结果
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'learning_rate': lr, 'hidden_size': hidden_size, 'reg_strength': reg_strength}

                # 记录每种参数组合下的最佳验证准确率
                best_acc_per_params[(lr, hidden_size, reg_strength)] = acc

    return best_acc, best_params, best_acc_per_params


