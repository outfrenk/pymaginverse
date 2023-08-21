import numpy as np


def reject_data(residual: np.ndarray,
                types_sort: np.ndarray,
                rej_crits: np.ndarray
                ) -> np.ndarray:
    if rej_crits.shape != (7,) and rej_crits.shape != (7, len(residual[0])):
        raise Exception('rejection criteria incorrectly inputted!'
                        'should be an array with 7 rows with optional columns'
                        'for time dependence. Shape is now: ', rej_crits.shape)
    if len(types_sort) != len(residual):
        raise Exception(f'length types array does not match length residual; '
                        f'{len(types_sort)} vs {len(residual)}')
    type06 = types_sort % 7
    rejection_matrix = rej_crits[type06]
    if rej_crits.shape == (7,):
        rejection_matrix = np.repeat(rejection_matrix, len(residual[0])
                                     ).reshape(len(residual), -1)
    # if small enough 1, otherwise 0
    accept_matrix = np.where(residual <= rejection_matrix, 1, 0)

    return accept_matrix
