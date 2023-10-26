## Problem

You want to solve [Lasso regression][1] but know only [Gradient Descent][2] and can't compute gradient of lasso regularizer.

## Solution

* [Proximal Gradient][3] (PG) descent
* Accelerated Proximal Gradient (APG) descent (Nesterov's)
* Accelerated Proximal Gradient with Restart (APGRestart) descent
* [Coordinate descent][4] (CD)

## Optimizations

* Running delta: instead of computing fresh delta at every iteration, compute incrementally by difference.
* matrix.dot(sparse_vector): filter out zero rows of the vector

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
[2]: https://en.wikipedia.org/wiki/Gradient_descent
[3]: https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning#Lasso_regularization
[4]: https://en.wikipedia.org/wiki/Coordinate_descent
