import torch
from denseflow.utils import sum_except_batch
from denseflow.transforms.bijections.functional.mixtures import gaussian_mixture_transform, logistic_mixture_transform, censored_logistic_mixture_transform
from denseflow.transforms.bijections.functional.mixtures import get_mixture_params
from denseflow.transforms.bijections.conditional.autoregressive import ConditionalAutoregressiveBijection2d


class ConditionalGaussianMixtureAutoregressiveBijection2d(ConditionalAutoregressiveBijection2d):

    def __init__(self, autoregressive_net, num_mixtures, autoregressive_order='raster_cwh'):
        super(ConditionalGaussianMixtureAutoregressiveBijection2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = gaussian_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class ConditionalLogisticMixtureAutoregressiveBijection2d(ConditionalAutoregressiveBijection2d):

    def __init__(self, autoregressive_net, num_mixtures, autoregressive_order='raster_cwh'):
        super(ConditionalLogisticMixtureAutoregressiveBijection2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = logistic_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class ConditionalCensoredLogisticMixtureAutoregressiveBijection2d(ConditionalAutoregressiveBijection2d):

    def __init__(self, autoregressive_net, num_mixtures, num_bins, autoregressive_order='raster_cwh'):
        super(ConditionalCensoredLogisticMixtureAutoregressiveBijection2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_mixtures = num_mixtures
        self.num_bins = num_bins
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = censored_logistic_mixture_transform(inputs=inputs,
                                                logit_weights=logit_weights,
                                                means=means,
                                                log_scales=log_scales,
                                                num_bins=self.num_bins,
                                                eps=self.eps,
                                                max_iters=self.max_iters,
                                                inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)
