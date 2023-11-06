import numpy as np
import pandas as pd

def find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):
    """Find the closest number in the array from a given number x.

    Parameters
    ----------
    closest_to : float
        The target number(s) to find the closest of.
    list_to_search_in : list
        The list of values to look in.
    direction : str
        "both" for smaller or greater, "greater" for only greater numbers and "smaller" for the closest smaller.
    strictly : bool
        False for stricly superior or inferior or True for including equal.
    return_index : bool
        If True, will return the index of the closest value in the list.

    Returns
    ----------
    closest : int
        The closest number in the array.

    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Single number
    >>> x = nk.find_closest(1.8, [3, 5, 6, 1, 2])
    >>> x  #doctest: +SKIP
    >>>
    >>> y = nk.find_closest(1.8, [3, 5, 6, 1, 2], return_index=True)
    >>> y  #doctest: +SKIP
    >>>
    >>> # Vectorized version
    >>> x = nk.find_closest([1.8, 3.6], [3, 5, 6, 1, 2])
    >>> x  #doctest: +SKIP

    References
    -----------
    - Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lesspinasse, F., Pham, H., Schölzel, C., & S H Chen, A. (2020). NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing. Retrieved March 28, 2020, from https://github.com/neuropsychology/NeuroKit
    - MIT License. Copyright (c) 2020, Dominique Makowski

    - Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions: The above copyright notice and 
    this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    """

    # Transform to arrays
    closest_to = np.asarray(closest_to).reshape(-1,)
    list_to_search_in = pd.Series(np.array(list_to_search_in))

    out = [_find_closest(i, list_to_search_in, direction, strictly, return_index) for i in closest_to]

    if len(out) == 1:
        return out[0]
    else:
        return np.array(out)



# =============================================================================
# Internal
# =============================================================================
def _find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):

    try:
        index, closest = _find_closest_single_pandas(closest_to, list_to_search_in, direction, strictly)
    except ValueError:
        index, closest = np.nan, np.nan

    if return_index is True:
        return index
    else:
        return closest


# =============================================================================
# Methods
# =============================================================================


def _findclosest_base(x, vals, direction="both", strictly=False):
    if direction == "both":
        closest = min(vals, key=lambda y: np.abs(y - x))
    if direction == "smaller":
        if strictly is True:
            closest = max(y for y in vals if y < x)
        else:
            closest = max(y for y in vals if y <= x)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda y: y > x, vals))
        else:
            closest = min(filter(lambda y: y >= x, vals))

    return closest


def _find_closest_single_pandas(x, vals, direction="both", strictly=False):

    if direction in ["both", "all"]:
        index = (np.abs(vals - x)).idxmin()

    if direction in ["smaller", "below"]:
        if strictly is True:
            index = (np.abs(vals[vals < x] - x)).idxmin()
        else:
            index = (np.abs(vals[vals <= x] - x)).idxmin()

    if direction in ["greater", "above"]:
        if strictly is True:
            index = (vals[vals > x] - x).idxmin()
        else:
            index = (vals[vals >= x] - x).idxmin()

    closest = vals[index]

    return index, closest
