import torch
import torch.nn as nn

# output time_seq = (time_seq + 2*pad - dil*(kernel_size-1) -1) / stride + 1

class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_factor):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.kernel_size = kernel_size

        self.dilated_conv1d = nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             dilation=dilation_factor)
        self.dilated_conv1d.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         dilation=dilation_factor)
        self.skip_connection.apply(weights_init)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.leaky_relu(self.dilated_conv1d(x))
        x2 = self.skip_connection(x)
        # |x2| = (batch_size, out_channels, input_time_seq)
        x2 = x2[:, :, self.dilation_factor*(self.kernel_size-1):]
        # |x2| = |x1|
        return x1 + x2

class CNNForecasting(nn.Module):
    def __init__(self, config):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        self.dilation_factors = [2 ** i for i in range(0, config.num_layers-1)]
        self.dilation_factors.append(3)
        
        self.dilated_convs = nn.ModuleList(
            [DilatedConv1d(
                config.in_channels[i],
                config.out_channels[i],
                config.kernel_size[i],
                self.dilation_factors[i]
                ) for i in range(config.num_layers)]
        )

        for dilated_conv in self.dilated_convs:
            dilated_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=config.out_channels[-1],
                                      out_channels=1,
                                      kernel_size=1)
        self.output_layer.apply(weights_init)

    def forward(self, x):
        for dilated_conv in self.dilated_convs:
            x = dilated_conv(x)
        # |x| = (batch_size, config.out_channels[-1], 1)
        x = self.output_layer(x)
        # |x| = (batch_size, 1, 1)
        x = x.reshape(-1,1)
        return x