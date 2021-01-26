import torch

N, D_in, H, D_Out = 64, 100, 10, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4

# optim 패키지를 사용해 가중치를 계산하는 optimizer를 정의. Adam의 첫 번째 인자는 어떤 Tensor가 갱신되어야 하는지 알려준다.
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for t in range(500):
    print(t, loss.item())

optimizer.zero_grad()

loss.backward()

# autograd_tutorial과 다른 부분. optimizer가 매개변수를 갱신시키고 있다.
optimizer.step()
