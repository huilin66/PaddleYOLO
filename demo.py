import paddle

# 创建形状为[1,0,1]的空Tensor
tensor1 = paddle.empty([1,0,1])
tensor2 = paddle.empty([1,0,1])
# ... 创建更多的空Tensor

# 使用paddle.concat进行合并
result = paddle.concat([tensor1, tensor2], axis=0)  # 在第一个维度上进行合并

print(result.shape)  # 输出：[2, 0, 1]
tensors = paddle.repeat_interleave(tensor2, 3, axis=0)
print(tensors.shape)