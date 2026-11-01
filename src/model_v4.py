"""Persistable feature slicing for V4 heads; predictions consume one common descriptor."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FeatureHead(ClassifierMixin, BaseEstimator):
    def __init__(self, model, start=0, stop=924, input_dim=1748):
        self.model=model; self.start=start; self.stop=stop; self.input_dim=input_dim
        self.classes_=np.array([0,1]); self.n_features_in_=input_dim

    def fit(self,X,y=None):
        # sklearn's fitted-estimator checks require a fit method. This adapter
        # wraps an already fitted head and must only be used via FrozenEstimator.
        raise RuntimeError('FeatureHead is a frozen adapter; fit the underlying head first')

    def predict_proba(self,X):
        X=np.asarray(X)
        if X.ndim!=2 or X.shape[1]!=self.input_dim: raise ValueError('Wrong common feature width')
        return self.model.predict_proba(X[:,self.start:self.stop])

    def predict(self,X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


class GlobalPadAdapter(ClassifierMixin, BaseEstimator):
    """Exact packaging adapter for a selected GLOBAL-only frozen V4 head.

    All fitted global classifiers ignore columns 924:1748. This adapter pads those
    unused columns with zeros, so deployment need not extract local features.
    Packaging verifies exact predictions on every already locked test row.
    """
    def __init__(self,model):
        self.model=model; self.classes_=np.array([0,1]); self.n_features_in_=924

    def fit(self,X,y=None):
        raise RuntimeError('Packaging adapter is frozen')

    def predict_proba(self,X):
        X=np.asarray(X)
        if X.ndim!=2 or X.shape[1]!=924: raise ValueError('Global-only head expects 924 dimensions')
        return self.model.predict_proba(np.concatenate([X,np.zeros((len(X),824),dtype=X.dtype)],axis=1))

    def predict(self,X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
