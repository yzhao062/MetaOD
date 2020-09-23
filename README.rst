Automating Outlier Detection via Meta-Learning (MetaOD)
=====================================================================

**Development Status**: **As of 09/22/2020, MetaOD is under active development and in its alpha stage. Please follow, star, and fork to get the latest update**! 
For paper reproducibility, please see the paper_reproducibility folder for instruction.

**Given an unsupervised outlier detection (OD) task on a new dataset, how can we automatically select a good outlier detection method and its hyperparameter(s) (collectively called a model)?**
Thus far, model selection for OD has been a "black art"; as any model evaluation is infeasible due to the lack of (i) hold-out data with labels, and (ii) a universal objective function.

**In this work, we develop the first principled data-driven approach to model selection for OD, called MetaOD, based on meta-learning**.
In short, MetaOD is trained on extensive OD benchmark datasets to capitalize the prior experience so that **it could select the potentially best performing model for unseen datasets**.
*Simply put, one could feed in a dataset, and MetaOD will return the potentially best outlier detection model for it*, which boosts both detection quality and reduces the cost of running multiple models .


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
* `Quick Start for Meta Feature Generation <#quick-start-for-meta-feature-generation>`_


------------

System Introduction
^^^^^^^^^^^^^^^^^^^

As shown in the figure below, MetaOD contains offline meta-learner training and online model selection.
For selecting an outlier detection model for a new dataset, one only needs the online model selection. Specifically, to be finished.


.. image:: https://raw.githubusercontent.com/yzhao062/MetaOD/master/docs/images/MetaOD_Flowchart.jpg
   :target: https://raw.githubusercontent.com/yzhao062/MetaOD/master/docs/images/MetaOD_Flowchart.jpg
   :alt: metaod_flow
   :align: center

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


Quick Start for Meta Feature Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Getting the embedding of an arbitrary dataset is first step of MetaOD, which
cam be done by our specialized meta-feature generation function.

It may be used for other purposes as well, e.g., measuring the similarity of
two datasets.

.. code-block:: python

    # import meta-feature generator
    from metaod.models.gen_meta_features import gen_meta_features

    meta_features = gen_meta_features(X)

A simple example of visualizing two different environments using TSNE with
our meta-features are shown below. The environment on the left is composed
100 datasets with similarity, and the same color stands for same group of datasets.
The environment on the left is composed
62 datasets without known similarity. Our meta-features successfully capture
the underlying similarity in the left figure.

.. image:: https://raw.githubusercontent.com/yzhao062/MetaOD/master/docs/images/meta_vis.jpg
   :target: https://raw.githubusercontent.com/yzhao062/MetaOD/master/docs/images/meta_vis.jpg
   :alt: meta_viz
   :align: center
