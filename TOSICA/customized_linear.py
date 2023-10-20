#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From Uchida Takumi https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
extended torch.nn module which cusmize connection.
This code base on https://pytorch.org/docs/stable/notes/extending.html
"""
import torch
import torch.nn as nn
import torch_sparse
import math

class SparseLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        sparsity: sparsity of weight matrix
            Default: 0.9
        connectivity: user defined sparsity matrix
            Default: None
        small_world: boolean flag to generate small world sparsity
            Default: ``False``
        dynamic: boolean flag to dynamically change the network structure
            Default: ``False``
        deltaT (int): frequency for growing and pruning update step
            Default: 6000
        Tend (int): stopping time for growing and pruning algorithm update step
            Default: 150000
        alpha (float): f-decay parameter for cosine updates
            Default: 0.1
        max_size (int): maximum number of entries allowed before chunking occurrs
            Default: 1e8

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.SparseLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features, out_features, indices, bias=True):
        assert in_features < 2 ** 31 and out_features < 2 ** 31
        assert isinstance(indices, torch.LongTensor) or isinstance(indices,torch.cuda.LongTensor), "Connectivity must be a Long Tensor"
        assert indices.shape[0] == 2 and indices.shape[1] > 0, "Input shape for connectivity should be (2,nnz)"
        assert indices.shape[1] <= in_features * out_features, "Nnz can't be bigger than the weight matrix"
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        nnz = indices.shape[1]
        self.sparsity = nnz / (out_features * in_features)

        values = torch.empty(nnz)

        indices, values = torch_sparse.coalesce(indices, values, out_features, in_features)

        self.register_buffer('indices', indices)
        self.weights = nn.Parameter(values)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        nn.init.uniform_(self.weights, -stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @property
    def weight(self):
        """ returns a torch.sparse.FloatTensor view of the underlying weight matrix
            This is only for inspection purposes and should not be modified or used in any autograd operations
        """
        weight = torch.sparse.FloatTensor(self.indices, self.weights, (self.out_features, self.in_features))
        return weight.coalesce().detach()

    def forward(self, inputs):

        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features

        if len(output_shape) == 1: inputs = inputs.view(1, -1)
        inputs = inputs.flatten(end_dim=-2)

        output = torch_sparse.spmm(self.indices, self.weights, self.out_features, self.in_features, inputs.t()).t()
        if self.bias is not None:
            output += self.bias

        return output.view(output_shape)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparsity={}, connectivity={}, small_world={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sparsity, self.connectivity,
            self.small_world
        )


class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Args:
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )





if __name__ == 'check grad':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    customlinear = CustomizedLinearFunction.apply

    input = (
            torch.randn(20,20,dtype=torch.double,requires_grad=True),
            torch.randn(30,20,dtype=torch.double,requires_grad=True),
            None,
            None,
            )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)
