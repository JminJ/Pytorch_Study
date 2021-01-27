import random 
import torch
import torch.nn as nn

class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = nn.Linear(D_in, H)
        self.middle_linear = nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min = 0)

        # 0-3까지 한 가지 수를 골라 범위로 지정
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min = 0)

        y_pred = self.output_linear(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

crit = nn.MSELoss(reduction='sum')

# 이 모델은 매우 이상한 모델이므로 SGD를 이용해 학습하는 것은 어려우므로 momentum을 사용한다.
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)
for t in range(500):
    y_pred = model(x)

    loss = crit(ypred, y)
    if t % 100 == 99:
        print(t, loss.item())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


