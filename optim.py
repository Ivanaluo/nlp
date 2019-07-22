# SGD 就是随机梯度下降
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
# momentum 动量加速,在SGD函数里指定momentum的值即可
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
'''
SGD每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程
Momentum 传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 此方法比较曲折。
'''


'''
AdaGrad 优化学习率，使得每一个参数更新都会有自己与众不同的学习率。
与momentum类似，不过不是给喝醉酒的人安排另一个下坡, 而是给他一双不好走路的鞋子, 
使得他一摇晃着走路就脚疼, 鞋子成为了走弯路的阻力, 逼着他往前直着走.
'''
# RMSprop 指定参数alpha
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# Adam 参数betas=(0.9, 0.99)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
'''
RMSProp 有了 momentum 的惯性原则 , 加上 adagrad 的对错误方向的阻力, 我们就能合并成这样. 
让 RMSProp同时具备他们两种方法的优势. 不过细心的同学们肯定看出来了, 似乎在 RMSProp 中少了些什么. 
原来是我们还没把 Momentum合并完全, RMSProp 还缺少了 momentum 中的 这一部分. 
所以, 我们在 Adam 方法中补上了这种想法. 
Adam 计算m 时有 momentum 下坡的属性, 计算 v 时有 adagrad 阻力的属性, 然后再更新参数时 把 m 和 V 都考虑进去. 
实验证明, 大多数时候, 使用 adam 都能又快又好的达到目标, 迅速收敛. 
所以说, 在加速神经网络训练的时候, 一个下坡, 一双破鞋子, 功不可没. 
'''



# 为每个优化器创建一个 net
net_SGD         = Net()
net_Momentum    = Net()
net_RMSprop     = Net()
net_Adam        = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):

        # 对每个优化器, 优化属于他的神经网络
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)              # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            l_his.append(loss.data.numpy())     # loss recoder

