- kan的实现基于efficient-kan(https://github.com/Blealtan/efficient-kan)
- 开发板为RK3566，使用rknntoolkit2-2.0.0版本

## 使用说明

### pc端
- 模型训练

运行train.py将训练一个模型并保存在cp目录下
- 模型导出

运行export_onnx.py将导出kan.onnx

### 板端
将rk目录下的文件放至开发板，注意修改其中的设备类型和设备ip，理论上rknntoolkit2的设备均可
- 模型转换

运行convert_kan.py，将得到kan.rknn模型
- 模型推理

运行infer.py，将会得到如下输出，对应图片的‘6’（可以运行show_test.py查看测试图片）
- 输出
```commandline
[array([[-23.453125  , -11.1796875 , -19.46875   ,  -8.        ,
        -45.6875    , -17.765625  ,  -0.69189453, -20.5       ,
        -34.9375    , -49.        ]], dtype=float32)]
```


## 转换说明
只遇到了一个广播相关的问题，具体的代码修改见kan.py-93行
原代码为:
```python
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    
```
修改后为:
```python
        x = x.unsqueeze(-1)
        tempg = grid.unsqueeze(0).expand(x.size(0),-1,-1)
        xtemp = x.expand(-1,-1,tempg.size(-1)-1)
        t1 = xtemp >= grid[:, :-1]
        t2 = xtemp < grid[:, 1:]
        bases = (t1 & t2).to(x.dtype)
```
