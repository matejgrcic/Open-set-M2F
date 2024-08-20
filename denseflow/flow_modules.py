import torch
import torch.nn as nn
from denseflow.transforms import Transform, Squeeze2d, Slice, WaveletSqueeze2d, SwitchChannels
from denseflow.transforms import Conv1x1, BatchNormBijection2d, ActNormBijection2d
from denseflow.distributions import StandardNormal
from denseflow.nn.layers import ElementwiseParams2d
from denseflow.nn.nets import DenseNet
from denseflow.utils import sum_except_batch
import torch.nn.functional as F

from .affine_coupling import SingleAffineCoupling

class SequentialTransform(Transform):
    def __init__(self, transforms):
        super(SequentialTransform, self).__init__()
        assert type(transforms) == torch.nn.ModuleList
        self.transforms = transforms

    def forward(self, x):
        total_ld = torch.zeros(x.shape[0]).to(x)
        for layer in self.transforms:
            x, ld = layer(x)
            total_ld += ld
        return x, total_ld

    def inverse(self, z):
        for layer in reversed(self.transforms):
            z = layer.inverse(z)
        return z

class _BNReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_BNReLUConv, self).__init__()
        self.transformation = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * in_channels, 2 * out_channels, kernel_size=1))
        nn.init.zeros_(self.transformation[-1].weight)
        nn.init.zeros_(self.transformation[-1].bias)

    def forward(self, x):
        mu, unconstrained_scale = self.transformation(x).chunk(2, dim=1)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        return mu, log_scale
        # scale = 0.1 * torch.tanh(0.5 * unconstrained_scale) + 1
        # return mu, scale



class InvertibleTransition(Transform):

    def __init__(self, in_channels):
        super(InvertibleTransition, self).__init__()
        # self.transition_conv = Conv1x1(in_channels)
        self.squeeze = Squeeze2d()
        # self.squeeze = WaveletSqueeze2d(in_channels)
        self.dist = StandardNormal((in_channels * 2, 2, 2))
        # self.bn = ActNormBijection2d(in_channels)

    def forward(self, x):
        # x, ld1 = self.bn(x)
        # x, ld2 = self.transition_conv(x)
        x, _ = self.squeeze(x)

        x1, x2 = torch.chunk(x, 2, dim=1)

        logpz = self.dist.log_prob(x2)

        # return x2, ld2 + ld1 + logpz
        return x1, logpz


    def inverse(self, z1):
        z2 = self.dist.sample(z1.shape)
        z = torch.cat([z1, z2], dim=1)
        z = self.squeeze.inverse(z)
        return z
        # z = self.transition_conv.inverse(z)
        # return z
        # return self.bn.inverse(z)

class InvertibleDenseBlock(Transform):
    stochastic_forward = True

    def __init__(self, in_channels, num_steps, num_layers, layer_mid_chnls, growth_rate=None, checkpointing=False):
        super(InvertibleDenseBlock, self).__init__()

        self.num_steps = num_steps
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        self.noise_params = nn.ModuleList([])

        blocks = []
        noise_in_chnls = self.compute_noise_inputs(in_channels, growth_rate, num_steps)
        for i in range(self.num_steps):
            chnls = int(in_channels + i * growth_rate)
            if i != self.num_steps - 1:
                self.noise_params.append(_BNReLUConv(noise_in_chnls[i], self.growth_rate))

            # ac = AdditiveCoupling(
            layers = [
                SequentialTransform(nn.ModuleList([
                    ActNormBijection2d(chnls),
                    Conv1x1(chnls),
                    # SwitchChannels(),
                    SingleAffineCoupling(chnls, mid_chnls=layer_mid_chnls,checkpointing=checkpointing),
                    SwitchChannels()
                    # ImprovedCoupling(chnls, mid_chnls=layer_mid_chnls, checkpointing=checkpointing)
                ])) for _ in range(num_layers)]
            # layers = [AffineCoupling(chnls, mid_chnls=layer_mid_chnls,checkpointing=checkpointing) for _ in range(num_layers)]
            blocks.append(SequentialTransform(nn.ModuleList(layers)))
        self.blocks = nn.Sequential(*blocks)
        self.noise_dist = torch.distributions.Normal(0, 1)

    def compute_noise_inputs(self, in_c, gr, steps):
        inputs = []
        out = []
        for i in range(steps):
            if i != 0:
                inputs.append(in_c)
                in_c += gr
            else:
                inputs.append(in_c)
            out.append(sum(inputs))
        return out


    def forward(self, x):
        total_ld = torch.zeros(x.shape[0]).to(x)
        N, _, H, W = x.shape
        noisy_inputs = []
        x_in = x
        for i, block in enumerate(self.blocks.children()):
            if i != 0:
                x_in_noisy = torch.cat(noisy_inputs, dim=1)
                eta = self.noise_dist.sample(torch.Size([N, self.growth_rate, H, W])).to(x)
                mu, log_scale = self.noise_params[i-1](x_in_noisy)
                eta_hat = mu + torch.exp(log_scale) * eta
                log_p = self.noise_dist.log_prob(eta)
                total_ld -= log_p.sum(dim=(1, 2, 3))
                total_ld += sum_except_batch(log_scale)
                noisy_inputs.append(x_in)
                x_in = torch.cat((x_in, eta_hat), dim=1)
                x_out, ldi = block(x_in)
                x_in = x_out
            else:
                noisy_inputs.append(x_in)
                x_out, ldi = block(x_in)
                x_in = x_out

            total_ld = total_ld + ldi
        return x_out, total_ld

    def inverse(self, z):
        # block = list(self.blocks.children())[-1]
        # z = block.inverse(z)
        # z = z[:, :self.in_channels]

        # total_z = []
        for i, block in enumerate(reversed(list(self.blocks.children()))):

            z = block.inverse(z)
            if i != self.num_steps - 1:
                z = z[:, :-self.growth_rate]
            # else:
            #     total_z.append(z)
        # z = sum(total_z) / len(total_z)
        return z