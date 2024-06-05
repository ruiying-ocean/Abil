import warnings
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                      indexable)

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import ConformityScore


class OffsetGammaConformityScore(ConformityScore):
    """
    Gamma conformity score.

    The signed conformity score = (y - y_pred) / y_pred.
    The conformity score is not symmetrical.

    This is appropriate when the confidence interval is not symmetrical and
    its range depends on the predicted values. Like the Gamma distribution,
    its support is limited to strictly positive reals.
    """

    def __init__(
        self,
        sym: bool = False,
        offset=0,
    ) -> None:
        super().__init__(sym=sym, consistency_check=False, eps=EPSILON)
        self.offset = offset  # Adding a new instance variable 'new_variable' initialized to None

    def _check_observed_data(
        self,
        y: ArrayLike,
    ) -> None:
        if not self._all_non_negative(y):
            raise ValueError(
                f"At least one of the observed target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be non-negative, "
                "in conformity with the offset Gamma distribution support."
            )

    def _check_predicted_data(
        self,
        y_pred: ArrayLike,
    ) -> None:
        if not self._all_non_negative(y_pred):
            raise ValueError(
                f"At least one of the predicted target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be non-negative, "
                "in conformity with the offset Gamma distribution support."
            )

    @staticmethod
    def _all_non_negative(
        y: ArrayLike,
    ) -> bool:
        return np.all(np.greater_equal(y, 0))

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the observed values
        and the predicted ones, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        """
        self._check_observed_data(y)
        self._check_predicted_data(y_pred)
        return np.divide(np.subtract(y, y_pred), y_pred+self.offset)

    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        <=> y = y_pred * (1 + signed conformity score)

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        self._check_predicted_data(y_pred)
        return np.multiply(y_pred, np.add(1, conformity_scores))
