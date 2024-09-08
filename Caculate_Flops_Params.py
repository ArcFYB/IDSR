# pip install thop

from thop import profile
import torch
import torchvision
import numpy as np

if __name__ == "__main__":
    # 加载预训练模型
    pretrained_model_path = '/home/fiko/Code/Super_Resolution/End2End_SR/experiments/loss1__loss1+loss2/resume/I190000_E2090_gen.pth'
    pretrained_dict = torch.load(pretrained_model_path)

    # 实例化你的模型 或者 放到inference过程
    model = torchvision.models.alexnet(pretrained=False)
    # model = UNet()

    # 获取模型的当前状态字典
    model_dict = model.state_dict()

    # 将预训练参数加载到模型中
    # 注意：确保预训练模型的键和你的模型的键匹配
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 设置模型为评估模式
    model.eval()
    # noise  = model(torch.cat([x_SR, x_noisy], dim=1),time=sqrt_alpha_cumprod,step_t = 100)
    
    
    # -------------------------------------------------------------------------------------------------
    # 方法一 调用thop库
    dummy_input = torch.randn(1, 6, 256, 256)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # -------------------------------------------------------------------------------------------------
    # 方法二 手动计算
    # 遍历model.parameters()返回的全局参数列表
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量
            
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')    # 遍历model.parameters()返回的全局参数列表
    # -------------------------------------------------------------------------------------------------