import numpy as np

def get_max_len(array):
    max_value = 0

    for arr in array:
        max_value = max(max_value, len(arr))

    return max_value


def pad_sequences(
    seqs: list,
    pad_value: int = 0,
    max_len: int | None = None
) -> np.ndarray:

    if len(seqs) == 0:
        return np.empty((0, 0), dtype=np.int64)

    output = []
    max_array_value = get_max_len(seqs)

    if max_len is not None:
        max_array_value = max_len

    for arr in seqs:
        result = arr[:max_array_value] + [
            pad_value
        ] * max(0, max_array_value - len(arr))

        output.append(result)

    return np.array(output, dtype=np.int64)