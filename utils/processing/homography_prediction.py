import math
from abc import ABC, abstractclassmethod
from typing import Any, Union

import numpy as np
import torch


class HomographyPrediction(ABC):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.predict(*args, **kwds)

    @abstractclassmethod
    def predict(self, *args: Any, **kwds: Any) -> Any:
        ...


class TaylorHomographyPrediction(HomographyPrediction):
    def __init__(self, order: int = 2) -> None:
        r"""Computes the Taylor series expansion of a temporal
        function.

        Args:
            order (int): Order of Taylor series expansion.
        """
        self._order = order

    def predict(
        self, duvs: Union[np.ndarray, torch.Tensor], dt: float = 1.0
    ) -> Union[np.ndarray, torch.Tensor]:
        converted = False
        if isinstance(duvs, torch.Tensor):
            converted = True
            duvs = duvs.numpy()
        if len(duvs.shape) != 4:
            raise ValueError(
                f"Expected 4 dimensional input, got {len(duvs.shape)} dimensions."
            )
        duvs_pred = duvs[:, self._order :].copy()
        for order in range(1, self._order + 1):
            # torch.diff doesn't support higher orders
            diff_duv = np.diff(duvs, n=order, axis=1)[
                :, self._order - order :
            ]  # BxNx4x2 -> Bx(N-order)x4x2
            duvs_pred += diff_duv * dt**order / math.factorial(order)
        if converted:
            return torch.from_numpy(duvs_pred)
        return duvs_pred


class KalmanHomographyPrediction:
    def __init__(self) -> None:
        raise NotImplementedError("Not implemented.")

    def predict(self, duv: np.ndarray):
        raise NotImplementedError("Not implemented.")


if __name__ == "__main__":

    def test_taylor():
        order = 2
        N = 10

        duv = np.expand_dims(np.arange(0, N) ** order, [1, 2])
        duv = np.repeat(duv, 2, axis=-1)
        duv = np.repeat(duv, 4, axis=1)
        duv = np.expand_dims(duv, 0)
        duv = duv.astype(np.float32)

        duv = torch.from_numpy(duv)
        taylor = TaylorHomographyPrediction(order)
        duv_pred = taylor(duv)
        print("Future values:\n", duv[:, order + 1 :])
        print("Estimated future values:\n", duv_pred[:, :-1])
        print("Numerical error:\n", duv[:, order + 1 :] - duv_pred[:, :-1])

    def test_kalman():
        kalman = KalmanHomographyPrediction()
        kalman()

    test_taylor()
