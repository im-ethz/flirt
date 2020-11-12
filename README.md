# FLIRT
[![Python Versions](https://img.shields.io/pypi/pyversions/flirt.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/flirt/)
[![PyPI](https://img.shields.io/pypi/v/flirt.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/flirt/)
[![Documentation Status](https://readthedocs.org/projects/flirt/badge/?version=latest)](https://flirt.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/im-ethz/flirt/master)

![](https://github.com/im-ethz/flirt/raw/master/docs/img/flirt-header.png)

⭐️ **Simple. Robust. Powerful.** 

**FLIRT** is a **F**eature generation too**L**k**I**t for wea**R**able da**T**a such as that from your smartwatch or smart ring. With FLIRT you can
easily transform wearable data into meaningful features which can then be used for example in machine learning or AI models.

In contrast to other existing toolkits, FLIRT (1) focuses on physiological data recorded with
(consumer) **wearables** and (2) calculates features based on a **sliding-window approach**.
FLIRT is an easy-to-use, robust and efficient feature generation toolkit for your wearable device!

![FLIRT Workflow](https://github.com/im-ethz/flirt/raw/master/docs/img/flirt-workflow.png)

➡️ **Are you ready to FLIRT with your wearable data?**

## Main Features
A few things that FLIRT can do:
  - Loading data from common wearable device formats such as from the Empatica E4 or Holter ECGs
  - Overlapping sliding-window approach for feature calculation
  - Calculating [HRV](https://flirt.readthedocs.io/en/latest/api.html#module-flirt.hrv) (heart-rate variability) features from NN intervals (aka inter-beat intervals)
  - Deriving features for [EDA](https://flirt.readthedocs.io/en/latest/api.html#module-flirt.eda) (electrodermal activity)
  - Computing features for [ACC](https://flirt.readthedocs.io/en/latest/api.html#module-flirt.acc) (accelerometer)
  - Provide and prepare features in one comprehensive DataFrame, so that they can directly be used for further steps
    (e.g. training machine learning models)

😎 FLIRT provides **high-level** implementations for fast and easy utilization of feature generators
(see [flirt.simple](https://flirt.readthedocs.io/en/latest/api.html#module-flirt.simple)).

🤓 For advanced users, who wish to adapt algorithms and parameters do their needs, FLIRT also provides **low-level**
implementations.
They allow for extensive configuration possibilities in feature generation and the specification of which algorithms to
use for generating features.


## Installation
FLIRT is available from [PyPI](https://pypi.org/project/flirt/) and can be installed via pip:
```
pip install flirt
```

Alternatively, you can checkout the source code from the [GitHub repository](https://github.com/im-ethz/flirt):
```
git clone https://github.com/im-ethz/flirt
```


# Quick example
Generate a comprehensive set of features for an Empatica E4 data archive with a single line of code 🚀
```
import flirt
features = flirt.with_.empatica('./1234567890_A12345.zip')
```

Check out the [documentation](https://flirt.readthedocs.io/) and exemplary [Jupyter notebooks](https://github.com/im-ethz/flirt/tree/master/notebooks/).

# Roadmap
Things FLIRT will be able to do in the future:
  - [ ] Use FLIRT with Oura's smart ring and other consumer-grade wearable devices
  - [ ] Use FLIRT with Apple Health to derive meaningful features from long-term data recordings
  - [ ] Feature generation for additional sensor modalities such as: blood oxygen saturation, blood volume changes, respiration rate, and step counts

# Authors
Made with ❤️ at [ETH Zurich](https://im.ethz.ch).

Check out all [authors](https://github.com/im-ethz/flirt/tree/master/docs/authors.rst).

# FAQs
- **How does FLIRT distinguish from other physiological data processing packages such as neurokit?**  \
    While FLIRT works with physiological data like other packages, it places special emphasis on the inherent challenges
    of data processing obtained from (consumer) wearable devices such as smartwaches instead of professional,
    medical-grade recording devices such as ECGs or EEGs. As an example, when processing data from smartwatches, one
    could be confronted with inaccurate data, which needs artifact removal, or measurement gaps, which need to be
    dealt with.
