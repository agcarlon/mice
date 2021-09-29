Multi-iteration Stochastic Estimator
============

The multi-iteration stochastic estimator (MICE) is an estimator of gradients to be used in stochastic optimization. It uses control-variates to build a hierarchy of iterations, adptively sampling to keep the statistical variance below tolerance in an optimal fashion, cost-wise. The tolerance on the statistical error decreases proportionally to the square of the gradient norm, thus, SGD-MICE converges linearly in strongly convex L-smooth functions.
