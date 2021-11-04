import torch


def get_monotonicity(monotonicity, num_channels):
    if isinstance(monotonicity, (int, float)):
        if not monotonicity in (-1, 0, 1):
            raise ValueError("monotonicity must be one of -1, 0, +1")
        return monotonicity * torch.ones(num_channels)
    else:
        if not (isinstance(monotonicity, torch.Tensor) and list(monotonicity.shape) == [num_channels]):
            raise ValueError("monotonicity must be either an int or a tensor with shape [num_channels]")
        if not torch.all(
            torch.eq(monotonicity, 0) | torch.eq(monotonicity, 1) | torch.eq(monotonicity, -1)
        ).item():
            raise ValueError("monotonicity must be one of -1, 0, +1")
        return monotonicity.float()


class BasePWL(torch.nn.Module):
    def __init__(self, num_breakpoints):
        super(BasePWL, self).__init__()
        if not num_breakpoints >= 1:
            raise ValueError(
                "Piecewise linear function only makes sense when you have 1 or more breakpoints."
            )
        self.num_breakpoints = num_breakpoints

    def slope_at(self, x):
        dx = 1e-3
        return -(self.forward(x) - self.forward(x + dx)) / dx


def calibrate1d(x, xp, yp):
    """
    x: [N, C]
    xp: [C, K]
    yp: [C, K]
    """
    x_breakpoints = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((x.shape[0], 1, 1))], dim=2)
    num_x_points = xp.shape[1]
    sorted_x_breakpoints, x_indices = torch.sort(x_breakpoints, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points), torch.tensor(num_x_points - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_x_breakpoints, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_x_breakpoints, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points), torch.tensor(num_x_points - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(x.shape[0], -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x + 1e-7)
    return cand


class Calibrator(torch.nn.Module):
    def __init__(self, keypoints, monotonicity, missing_value=11.11):
        """
        Calibrates input to the output range of [-0.5*monotonicity, 0.5*monotonicity].
        The output is always monotonic with respect to the input.
        Recommended to use Adam for training. The calibrator is initalized as a straight line.

        value <= keypoint[0] will map to -0.5*monotonicity.
        value >= keypoint[-1] will map to 0.5*monotonicity.
        value == missing_value will map to a learnable value (within the standard output range).
        Each channel is independently calibrated and can have its own keypoints.
        Note: monotonicity and keypoints are not trainable, they remain fixed, only the calibration output at
        each keypoint is trainable.

        keypoints: tensor with shape [C, K], where K > 2
        monotonicity: tensor with shape [C]
        missing_value: float
        """
        super(Calibrator, self).__init__()
        xp = torch.tensor(keypoints, dtype=torch.float32)
        self.register_buffer("offset", xp[:, :1].clone().detach())
        self.register_buffer("scale", (xp[:, -1:] - self.offset).clone().detach())
        xp = (xp - self.offset) / self.scale
        self.register_buffer("keypoints", xp)
        self.register_buffer("monotonicity", torch.tensor(monotonicity, dtype=torch.float32).unsqueeze(0))
        self.missing_value = missing_value
        yp = xp[:, 1:] - xp[:, :-1]
        # [C, K - 1]
        self.yp = torch.nn.Parameter(yp, requires_grad=True)
        # [1, C]
        self.missing_y = torch.nn.Parameter(torch.zeros_like(xp[:, 0]).unsqueeze(0), requires_grad=True)

    def forward(self, x):
        """Calibrates input x tensor. x has shape [BATCH_SIZE, C]."""
        missing = torch.zeros_like(x) + torch.tanh(self.missing_y) / 2.0
        yp = torch.cumsum(torch.abs(self.yp) + 1e-9, dim=1)
        xp = self.keypoints
        last_val = yp[:, -1:]
        yp = torch.cat([torch.zeros_like(last_val), yp / last_val], dim=1)
        x_transformed = torch.clamp((x - self.offset) / self.scale, 0.0, 1.0)
        calibrated = calibrate1d(x_transformed, xp, yp) - 0.5
        return self.monotonicity * torch.where(x == self.missing_value, missing, calibrated)


class BasePWLX(BasePWL):
    def __init__(self, num_channels, num_breakpoints, num_x_points):
        super(BasePWLX, self).__init__(num_breakpoints)
        self.num_channels = num_channels
        self.num_x_points = num_x_points
        # self.x_positions = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_x_points))
        self.x_positions = torch.Tensor(self.num_channels, self.num_x_points)
        self._reset_x_points()

    def _reset_x_points(self):
        # torch.nn.init.normal_(self.x_positions, std=0.000001)
        # torch.nn.init.zeros_(self.x_positions)
        self.x_positions = torch.linspace(-1,1,self.num_x_points).unsqueeze(0).expand(self.num_channels, self.num_x_points)

    def get_x_positions(self):
        return self.x_positions

    def get_sorted_x_positions(self):
        return torch.sort(self.get_x_positions(), dim=1)[0]

    def get_spreads(self):
        sorted_x_positions = self.get_sorted_x_positions()
        return (torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions)[:, :-1]

    def unpack_input(self, x):
        shape = list(x.shape)
        if len(shape) == 2:
            return x
        elif len(shape) < 2:
            raise ValueError(
                "Invalid input, the input to the PWL module must have at least 2 dimensions with channels at dimension dim(1)."
            )
        assert shape[1] == self.num_channels, (
            "Invalid input, the size of dim(1) must be equal to num_channels (%d)" % self.num_channels
        )
        x = torch.transpose(x, 1, len(shape) - 1)
        assert x.shape[-1] == self.num_channels
        return x.reshape(-1, self.num_channels)

    def repack_input(self, unpacked, old_shape):
        old_shape = list(old_shape)
        if len(old_shape) == 2:
            return unpacked
        transposed_shape = old_shape[:]
        transposed_shape[1] = old_shape[-1]
        transposed_shape[-1] = old_shape[1]
        unpacked = unpacked.view(*transposed_shape)
        return torch.transpose(unpacked, 1, len(old_shape) - 1)


class BasePointPWL(BasePWLX):
    def get_y_positions(self):
        raise NotImplementedError()

    def forward(self, x):
        old_shape = x.shape
        x = self.unpack_input(x)
        cand = calibrate1d(x, self.get_x_positions(), self.get_y_positions())
        return self.repack_input(cand, old_shape)


class PointPWL(BasePointPWL):
    def __init__(self, num_channels, num_breakpoints):
        super(PointPWL, self).__init__(num_channels, num_breakpoints, num_x_points=num_breakpoints + 1)
        self.y_positions = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_x_points))
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        with torch.no_grad():
            self.y_positions.copy_(self.get_sorted_x_positions())

    def get_x_positions(self):
        return self.x_positions

    def get_y_positions(self):
        return self.y_positions


class MonoPointPWL(BasePointPWL):
    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(MonoPointPWL, self).__init__(num_channels, num_breakpoints, num_x_points=num_breakpoints + 1)
        self.y_starts = torch.nn.Parameter(torch.Tensor(self.num_channels))
        self.y_deltas = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_breakpoints))
        self.register_buffer("monotonicity", get_monotonicity(monotonicity, num_channels))
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        with torch.no_grad():
            sorted_x_positions = self.get_sorted_x_positions()
            mono_mul = torch.where(
                torch.eq(self.monotonicity, 0.0),
                torch.tensor(1.0, device=self.monotonicity.device),
                self.monotonicity,
            )
            self.y_starts.copy_(sorted_x_positions[:, 0] * mono_mul)
            spreads = self.get_spreads()
            self.y_deltas.copy_(spreads * mono_mul.unsqueeze(1))

    def get_x_positions(self):
        return self.x_positions

    def get_y_positions(self):
        starts = self.y_starts.unsqueeze(1)
        deltas = torch.where(
            torch.eq(self.monotonicity, 0.0).unsqueeze(1),
            self.y_deltas,
            torch.abs(self.y_deltas) * self.monotonicity.unsqueeze(1),
        )
        return torch.cat([starts, starts + torch.cumsum(deltas, dim=1)], dim=1)


class BaseSlopedPWL(BasePWLX):
    def get_biases(self):
        raise NotImplementedError()

    def get_slopes(self):
        raise NotImplementedError()

    def forward(self, x):
        old_shape = x.shape
        x = self.unpack_input(x)
        bs = x.shape[0]
        sorted_x_positions = self.get_sorted_x_positions().cuda()
        skips = torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions
        slopes = self.get_slopes()
        skip_deltas = skips * slopes[:, 1:]
        biases = self.get_biases().unsqueeze(1)
        cumsums = torch.cumsum(skip_deltas, dim=1)[:, :-1]

        betas = torch.cat([biases, biases, cumsums + biases], dim=1)
        breakpoints = torch.cat([sorted_x_positions[:, 0].unsqueeze(1), sorted_x_positions], dim=1)

        # find the index of the first breakpoint smaller than x
        # TODO(pdabkowski) improve the implementation
        s = x.unsqueeze(2) - sorted_x_positions.unsqueeze(0)
        # discard larger breakpoints
        s = torch.where(s < 0, torch.tensor(float("inf"), device=x.device), s)
        b_ids = torch.where(
            sorted_x_positions[:, 0].unsqueeze(0) <= x,
            torch.argmin(s, dim=2) + 1,
            torch.tensor(0, device=x.device),
        ).unsqueeze(2)

        selected_betas = torch.gather(betas.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids).squeeze(2)
        selected_breakpoints = torch.gather(
            breakpoints.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        selected_slopes = torch.gather(slopes.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids).squeeze(2)
        cand = selected_betas + (x - selected_breakpoints) * selected_slopes
        return self.repack_input(cand, old_shape)


class PWL(BaseSlopedPWL):
    r"""Piecewise Linear Function (PWL) module.

    The module takes the Tensor of (N, num_channels, ...) shape and returns the processed Tensor of the same shape.
    Each entry in the input tensor is processed by the PWL function. There are num_channels separate PWL functions,
    the PWL function used depends on the channel.

    The x coordinates of the breakpoints are initialized randomly from the Gaussian with std of 2. You may want to
    use your own custom initialization depending on the use-case as the optimization is quite sensitive to the
    initialization of breakpoints. As long as your data is normalized (zero mean, unit variance) the default
    initialization should be fine.

    Arguments:
        num_channels (int): number of channels (or features) that this PWL should process. Each channel
            will get its own PWL function.
        num_breakpoints (int): number of PWL breakpoints. Total number of segments constructing the PWL is
            given by num_breakpoints + 1. This value is shared by all the PWL channels in this module.
    """

    def __init__(self, num_channels, num_breakpoints):
        super(PWL, self).__init__(num_channels, num_breakpoints, num_x_points=num_breakpoints)
        self.slopes = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_breakpoints + 1))
        self.biases = torch.nn.Parameter(torch.Tensor(self.num_channels))
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        torch.nn.init.ones_(self.slopes)
        self.slopes.data[:,:(self.num_breakpoints + 1)//2] = 0.0
        print()
        with torch.no_grad():
            self.biases.copy_(torch.zeros_like(self.biases))


    def get_biases(self):
        return self.biases

    def get_x_positions(self):
        return self.x_positions

    def get_slopes(self):
        return self.slopes


class MonoPWL(PWL):
    r"""Piecewise Linear Function (PWL) module with the monotonicity constraint.

    The module takes the Tensor of (N, num_channels, ...) shape and returns the processed Tensor of the same shape.
    Each entry in the input tensor is processed by the PWL function. There are num_channels separate PWL functions,
    the PWL function used depends on the channel. Each PWL is guaranteed to have the requested monotonicity.

    The x coordinates of the breakpoints are initialized randomly from the Gaussian with std of 2. You may want to
    use your own custom initialization depending on the use-case as the optimization is quite sensitive to the
    initialization of breakpoints. As long as your data is normalized (zero mean, unit variance) the default
    initialization should be fine.

    Arguments:
        num_channels (int): number of channels (or features) that this PWL should process. Each channel
            will get its own PWL function.
        num_breakpoints (int): number of PWL breakpoints. Total number of segments constructing the PWL is
            given by num_breakpoints + 1. This value is shared by all the PWL channels in this module.
        monotonicity (int, Tensor): Monotonicty constraint, the monotonicity can be either +1 (increasing), 
            0 (no constraint) or -1 (decreasing). You can provide either an int to set the constraint 
            for all the channels or a long Tensor of shape [num_channels]. All the entries must be in -1, 0, +1.
    """

    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(MonoPWL, self).__init__(num_channels=num_channels, num_breakpoints=num_breakpoints)
        self.register_buffer("monotonicity", get_monotonicity(monotonicity, self.num_channels))
        with torch.no_grad():
            mono_mul = torch.where(
                torch.eq(self.monotonicity, 0.0),
                torch.tensor(1.0, device=self.monotonicity.device),
                self.monotonicity,
            )
            self.biases.copy_(self.biases * mono_mul)

    def get_slopes(self):
        return torch.where(
            torch.eq(self.monotonicity, 0.0).unsqueeze(1),
            self.slopes,
            torch.abs(self.slopes) * self.monotonicity.unsqueeze(1),
        )

