#  介绍

数据集下载地址：https://github.com/zalandoresearch/fashion-mnist

# 操作流程

Model.py文件中定义了模型、训练方法和读取数据方法。

ParameterSearch.py文件中定义了参数查找方法。

TrainModel.py是模型训练代码，可在其中修改数据读取地址、调整参数，最后直接运行该文件实现模型训练和可视化，并且会保存最佳模型权重。

Test.py是模型测验代码，在其中修改测试数据集读取地址，运行该代码会加载之前训练好的模型的权重，实现模型在测试集上的预测。
