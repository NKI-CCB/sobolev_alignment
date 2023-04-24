"""
Multi KRR approximation.

@author: Soufiane Mourragui

Scripts supporting the naive integration of several KRR. No gain is provided by such approach.
"""

import torch


class MultiKRRApprox:
    """
    Multi Kernel Ridge Regression approximation.

    This class contains a wrapper around KRRApprox to serialise the approximation of latent factors.
    Several experiments show that such approach does not yield any advantage.
    """

    def __init__(self):
        """Create instance."""
        self.krr_regressors = []

    def predict(self, X: torch.Tensor):
        """Predict latent factor values given a tensor."""
        prediction = [clf.transform(torch.Tensor(X)).detach().numpy() for clf in self.krr_regressors]
        prediction = torch.Tensor(prediction)
        prediction = torch.mean(prediction, axis=0)

        return prediction

    def transform(self, X: torch.Tensor):
        """Predict latent factor values given a tensor."""
        return self.predict(X)

    def anchors(self):
        """Return anchors."""
        return self.anchors_

    def process_clfs(self):
        """Process the different classifiers."""
        self.anchors_ = torch.cat([clf.anchors() for clf in self.krr_regressors])
        self.sample_weights_ = torch.cat([clf.sample_weights_ for clf in self.krr_regressors])
        self.sample_weights_ = 1 / len(self.krr_regressors) * self.sample_weights_
        self.kernel_ = self.krr_regressors[0].kernel_

    def add_clf(self, clf):
        """Add a classifier."""
        self.krr_regressors.append(clf)
