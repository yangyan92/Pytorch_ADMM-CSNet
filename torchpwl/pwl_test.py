import pytest
import torch
import numpy as np

from .pwl import PointPWL, MonoPointPWL, SlopedPWL, MonoSlopedPWL

TOLERANCE = 1e-4

torch.manual_seed(11)


def get_x(num_channels, batch_size=37, std=3.):
    return torch.Tensor(batch_size, num_channels).normal_(mean=0., std=std)


@pytest.mark.parametrize("pwl_module",
                         [PointPWL, MonoPointPWL, SlopedPWL, MonoSlopedPWL])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize("num_breakpoints", [1, 7, 11])
def test_pwl_init(pwl_module, num_channels, num_breakpoints):
    module = pwl_module(
        num_channels=num_channels, num_breakpoints=num_breakpoints)
    x = get_x(num_channels)
    y = module(x)


@pytest.mark.parametrize("pwl_module",
                         [SlopedPWL, MonoSlopedPWL, MonoPointPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 7])
def test_pwl_default_init_response(pwl_module, num_channels, num_breakpoints):
    module = pwl_module(
        num_channels=num_channels, num_breakpoints=num_breakpoints)
    x = get_x(num_channels)
    y = module(x)
    # Should initialize to y = x by default.
    expected_y = x
    assert torch.max(torch.abs(y - expected_y)) < TOLERANCE


@pytest.mark.parametrize("pwl_module", [MonoSlopedPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 7])
@pytest.mark.parametrize("monotonicity", [-1, 0, 1])
def test_pwl_default_init_mono_response(pwl_module, num_channels,
                                        num_breakpoints, monotonicity):
    module = pwl_module(
        num_channels=num_channels,
        num_breakpoints=num_breakpoints,
        monotonicity=monotonicity)
    x = get_x(num_channels)
    y = module(x)
    # Should initialize to y = x if monotonicity is 1 or 0, otherwise y = -x
    expected_y = x if monotonicity in (1, 0) else -x
    assert torch.max(torch.abs(y - expected_y)) < TOLERANCE


@pytest.mark.parametrize("pwl_module", [MonoSlopedPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 7])
def test_pwl_default_init_multi_mono_response(pwl_module, num_channels,
                                              num_breakpoints):
    monotonicity = torch.Tensor(num_channels).normal_(std=100).long() % 3 - 1
    module = pwl_module(
        num_channels=num_channels,
        num_breakpoints=num_breakpoints,
        monotonicity=monotonicity)
    x = get_x(num_channels)
    y = module(x)
    # Should initialize to y = x if monotonicity is 1 or 0, otherwise y = -x
    expected_y = torch.where(torch.eq(monotonicity, -1).unsqueeze(0), -x, x)
    assert torch.max(torch.abs(y - expected_y)) < TOLERANCE


@pytest.mark.parametrize("pwl_module", [SlopedPWL, MonoSlopedPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 7])
def test_pwl_gradient_flows(pwl_module, num_channels, num_breakpoints):
    module = pwl_module(
        num_channels=num_channels, num_breakpoints=num_breakpoints)
    x = get_x(num_channels)
    x.requires_grad = True
    y = module(x)
    torch.sum(y).backward()
    expected_grad = 1.
    assert torch.max(torch.abs(x.grad - expected_grad)) < TOLERANCE


@pytest.mark.parametrize("pwl_module", [SlopedPWL, MonoSlopedPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 7])
def test_pwl_sloped_correct_num_breakpoints(pwl_module, num_channels,
                                            num_breakpoints):
    module = pwl_module(
        num_channels=num_channels, num_breakpoints=num_breakpoints)
    assert list(module.get_sorted_x_positions().shape) == [
        num_channels, num_breakpoints
    ]


@pytest.mark.parametrize("pwl_module",
                         [SlopedPWL, MonoSlopedPWL, PointPWL, MonoPointPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 2, 3, 4])
def test_pwl_is_continous(pwl_module, num_channels, num_breakpoints):
    module = pwl_module(
        num_channels=num_channels, num_breakpoints=num_breakpoints)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_()
    x = torch.linspace(
        -4., 4., steps=10000).unsqueeze(1).expand(-1, num_channels)
    y = module(x)
    dy = torch.roll(y, shifts=-1, dims=0) - y
    dx = torch.roll(x, shifts=-1, dims=0) - x
    grad = dy / dx
    if isinstance(module, (PointPWL, MonoPointPWL)):
        allowed_grad = torch.max(4 / module.get_spreads())
    else:
        allowed_grad = 4
    assert torch.max(abs(grad)) < allowed_grad


@pytest.mark.parametrize("pwl_module", [SlopedPWL, MonoSlopedPWL])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_breakpoints", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "optimizer_fn",
    [
        #lambda params: torch.optim.SGD(params=params, lr=0.1, momentum=0.5),
        lambda params: torch.optim.Adam(params=params, lr=0.2),
    ])
def test_pwl_fits(pwl_module, num_channels, num_breakpoints, optimizer_fn):
    module = pwl_module(
        num_channels=num_channels, num_breakpoints=num_breakpoints)
    bs = 128
    opt = optimizer_fn(module.parameters())
    steps = 4000
    loss_ = 0
    desired_loss = 0.02
    for step in range(steps):
        x = torch.Tensor(np.random.normal(0, scale=2, size=(bs, num_channels)))
        expected_y = torch.Tensor(
            np.random.normal(0, scale=0.1, size=(bs, num_channels)) +
            np.where(x > 0.2, x, 0.2))

        y = module(x)
        loss = torch.mean((expected_y - y)**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(loss.item())
        loss_ = 0.8 * loss_ + loss.item() * 0.2
        if loss_ < desired_loss:
            break
    assert loss_ < desired_loss


@pytest.mark.parametrize("pwl_module", [SlopedPWL, MonoSlopedPWL])
@pytest.mark.parametrize("input_shape", [
    (11, 5, 7, 3),
    (11, 6),
    (11, 1),
    (11, 1, 1, 1),
    (5, 1, 2, 1),
    (5, 2, 2, 1),
    (5, 2, 2),
])
def test_input_packing(pwl_module, input_shape):
    num_channels = input_shape[1]
    b = pwl_module(num_channels=num_channels, num_breakpoints=2)
    inp = torch.Tensor(*input_shape).normal_()
    unpacked_inp = b.unpack_input(inp)
    assert unpacked_inp.shape[1] == num_channels
    assert len(unpacked_inp.shape) == 2
    inp_restored = b.repack_input(unpacked_inp, inp.shape)
    assert list(inp_restored.shape) == list(inp.shape)
    assert torch.max(torch.abs(inp_restored - inp)).item() < TOLERANCE
