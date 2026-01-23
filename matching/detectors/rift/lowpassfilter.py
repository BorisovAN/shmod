import numpy as np
from scipy import fftpack

def lowpassfilter(sze, cutoff, n):
    """
    Constructs a low-pass butterworth filter.

    Parameters:
    sze (tuple or int): Size of the filter to construct [rows, cols].
    cutoff (float): Cutoff frequency of the filter (0 - 0.5).
    n (int): Order of the filter (>= 1). Note that n is doubled in the calculation.

    Returns:
    numpy.ndarray: The constructed low-pass filter.

    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('cutoff frequency must be between 0 and 0.5')

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be an integer >= 1')

    if isinstance(sze, int):
        rows, cols = sze, sze
    else:
        rows, cols = sze

    # Set up X and Y matrices with ranges normalized to +/- 0.5
    if cols % 2:
        xrange = np.linspace(-(cols-1)/2, (cols-1)/2, cols, dtype=np.float32) / (cols-1)
    else:
        xrange = np.linspace(-cols/2, (cols/2-1), cols, dtype=np.float32) / cols

    if rows % 2:
        yrange = np.linspace(-(rows-1)/2, (rows-1)/2, rows, dtype=np.float32) / (rows-1)
    else:
        yrange = np.linspace(-rows/2, (rows/2-1), rows, dtype=np.float32) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x**2 + y**2)  # Matrix with every pixel = radius relative to centre

    f = fftpack.ifftshift(1.0 / (1.0 + (radius / cutoff)**(2*n)))

    return f