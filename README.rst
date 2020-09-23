Automating Outlier Detection via Meta-Learning (MetaOD)
=====================================================================

**Development Status**: **As of 09/22/2020, MetaOD is under active development and in its alpha stage. Please follow, star, and fork to get the latest update**! 
For paper reproducibility, please see the paper_reproducibility folder for experimental environment.

**Given an unsupervised outlier detection (OD) task on a new dataset, how can we automatically select a good outlier detection method and its hyperparameter(s) (collectively called a model)?**
Thus far, model selection for OD has been a "black art"; as any model evaluation is infeasible due to the lack of (i) hold-out data with labels, and (ii) a universal objective function.
In this work, we develop the first principled data-driven approach to model selection for OD, called MetaOD,
based on meta-learning. MetaOD capitalizes on the past performances of a large body of detection models on existing outlier detection benchmark datasets, and carries over this prior experience to automatically select an effective model to be employed on a new dataset.
To capture task similarity, we introduce specialized meta-features that quantify outlying characteristics of a dataset.
Through comprehensive experiments, we show the effectiveness of MetaOD in selecting a detection model that significantly outperforms the most popular outlier detectors (e.g., LOF and iForest) as well as various state-of-the-art unsupervised meta-learners while being extremely fast.
To foster reproducibility and further research on this new problem, we open-source our entire meta-learning system, benchmark environment, and testbed datasets.

**MetaOD is an unsupervised method for selecting OD models on an arbitrary dataset**. MetaOD is trained on extensive OD benchmark datasets to capitalize the prior experiece; it could select the potentially best performing dataset for your datasets. *Simply put, you could plug in your dataset, and MetaOD will return the potentially best outlier detection model for you*!


`Preprint paper <https://arxiv.org/abs/2009.10606>`_ | `Reproducibility instruction <https://github.com/yzhao062/MetaOD/tree/master/paper_reproducibility>`_

**Citing MetaOD**\ :

If you use MetaOD in a scientific publication, we would appreciate
citations to the following paper::

    @article{zhao2020automating,
      author  = {Zhao, Yue and Ryan Rossi and Leman Akoglu},
      title   = {Automating Outlier Detection via Meta-Learning},
      journal = {arXiv preprint arXiv:2009.10606},
      year    = {2020},
    }

or::

    Zhao, Y., Rossi, R., and Akoglu, L., 2020. Automating Outlier Detection via Meta-Learning. arXiv preprint arXiv:2009.10606.
    
    
**Table of Contents**\ :


* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_

-----


Installation
^^^^^^^^^^^^

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as MetaOD is updated frequently:

.. code-block:: bash

   pip install metaod            # normal install
   pip install --upgrade metaod  # or update if needed
   pip install --pre metaod      # or include pre-release version for new features

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/metaod.git
   cd metaod
   pip install .
  
  
**Required Dependencies (to be cleaned)**\ :


* Python 3.5, 3.6, or 3.7
* joblib>=0.14.1
* liac-arff
* matplotlib
* numpy>=1.13
* scipy>=0.19.1
* scikit_learn>=0.19.1
* pandas
* psutil
* pyod>=0.7.5