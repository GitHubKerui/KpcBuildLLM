1 获取分类微调数据 
2 数据按比例切分，训练，验证，测试
3 数据Dataset实现 input 和 target的映射处理
3 数据转成tokenid并分批次
4 预训练模型的加载
5 最后一层linear的out设置成分类的数量
6 解冻最后的 linear，normal，transformer_block
7 训练观察验证
8 保存训练后模型。

