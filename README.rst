Automating Outlier Detection via Meta-Learning (MetaOD)
=====================================================================


.. image:: https://img.shields.io/pypi/v/metaod.svg?color=brightgreen
   :target: https://pypi.org/project/metaod/
   :alt: PyPI version

.. image:: https://img.shields.io/github/stars/yzhao062/metaod.svg
   :target: https://github.com/yzhao062/metaod/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/yzhao062/metaod.svg?color=blue
   :target: https://github.com/yzhao062/metaod/network
   :alt: GitHub forks

----

**Development Status**: **As of 09/26/2020, MetaOD is under active development and in its alpha stage. Please follow, star, and fork to get the latest update**!
For paper reproducibility, please see the paper_reproducibility folder for instruction.

**Given an unsupervised outlier detection (OD) task on a new dataset, how can we automatically select a good outlier detection method and its hyperparameter(s) (collectively called a model)?**
Thus far, model selection for OD has been a "black art"; as any model evaluation is infeasible due to the lack of (i) hold-out data with labels, and (ii) a universal objective function.
In this work, we develop the first principled data-driven approach to model selection for OD, called MetaOD, based on meta-learning.
In short, MetaOD is trained on extensive OD benchmark datasets to capitalize the prior experience so that **it could select the potentially best performing model for unseen datasets**.

Using MetaOD is easy.
**You could pass in a dataset, and MetaOD will return the most performing outlier detection models for it**, which boosts both detection quality and reduces the cost of running multiple models.


**API Demo for selecting outlier detection model on a new dataset (within 3 lines)**\ :


.. code-block:: python

   from metaod.models.utility import prepare_trained_model
   from metaod.models.predict_metaod import select_model

   # load pretrained MetaOD model
   prepare_trained_model()

   # use MetaOD to recommend models. It returns the top n model for new data X_train
   selected_models = select_model(X_train, n_selection=100)



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
* `Quick Start for Model Selection <#quick-start-for-model-selection>`_
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
  
  
**Required Dependencies**\ :


* Python 3.5, 3.6, or 3.7
* joblib>=0.14.1
* liac-arff
* numpy>=1.13
* scipy>=0.20
* **scikit_learn==0.22.1**
* pandas>=0.20
* pyod>=0.8

**Note**: Since we need to load trained models, we fix the scikit-learn version
to 0.20. We recommend you to use MetaOD in a fully fresh env to have the right dependency.


Quick Start for Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"examples/model_selection_example.py" <https://github.com/yzhao062/MetaOD/blob/master/examples/model_selection_example.py>`_
provide an example on using MetaOD for selecting top models on a new datasets, which is fully unsupervised.

The key procedures are below:

#. Load some synthetic datasets

.. code-block:: python

   # Generate sample data
   X_train, y_train, X_test, y_test = \
       generate_data(n_train=1000,
                     n_test=100,
                     n_features=3,
                     contamination=0.5,
                     random_state=42)

#. Use MetaOD to select top 100 models

.. code-block:: python

   from metaod.models.utility import prepare_trained_model
   from metaod.models.predict_metaod import select_model

   # load pretrained models
   prepare_trained_model()

   # recommended models. this returns the top model for X_train
   selected_models = select_model(X_train, n_selection=100)


#. Show the selected models' performance evaluation.

.. code-block:: python


   1st model Average Precision 0.9729833161334711
   10th model Average Precision 0.9631787029256742
   50th model Average Precision 0.9228434081007967
   100th model Average Precision 0.9228434081007967


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


