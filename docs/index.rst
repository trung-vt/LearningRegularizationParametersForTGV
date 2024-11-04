.. notebook_test documentation master file, created by
   sphinx-quickstart on Sat Jul 25 11:56:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. figure:: logo.jpg
..    :alt: Course Logo
..    :align: left
..    :width: 200px

Learning Regularization Parameters for TGV
===============================================================================


.. image:: https://img.shields.io/github/stars/borisbolliet/company_package?style=social
   :alt: GitHub stars
   :target: https://github.com/borisbolliet/company_package

.. image:: https://readthedocs.org/projects/company_package/badge/?version=latest
   :target: https://company_package.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/borisbolliet/company_package/blob/main/docs/tutorial.ipynb
   :alt: Open In Colab

| *Author*: Boris Bolliet


.. note::
    Package example for the
   `MPhil in Data Intensive Science <https://mphildis.bigdata.cam.ac.uk>`_ and the
   `MPhil in Economics and Data Science <https://www.econ.cam.ac.uk/postgraduate-studies/mphil-data>`_ at the University of Cambridge.


.. _Company_Package:

The **Company Package** is a Python package designed to model companies across different sectors.

Features
--------

- **Base `Company` class**: Core attributes and methods, including stock price retrieval.
- **Sector-specific subclasses**:

  - **InfoTechCompany**: For companies focused on information technology.
  - **FinTechCompany**: For companies in the financial technology sector.
  - **MedicalCompany**: With additional methods to track drug approval attempts.

- **Integration with `yfinance`**: Retrieves real-time stock information.

Installation
------------

Ensure you have Python 3.6 or higher. Install the package and its dependencies with:

.. code-block:: bash

    pip install -e .

Usage
-----

Here's a quick example of how to use the package:

.. code-block:: python

    import company as cp

    my_company = cp.Company(name="Nvidia", ticker="NVDA")
    my_company.display_info()

Documentation
-------------

Visit our `documentation page <https://your-readthedocs-url-here>`_.

Contributing
------------

Contributions are welcome! Fork our repository and submit a pull request.

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.



API Reference
-------------------------------------------------------------------------------


.. automodule:: company.base_company
    :members:
    :undoc-members:

.. automodule:: company.cli
    :members:
    :undoc-members:

.. automodule:: company.medical.medical
    :members:
    :undoc-members:






Useful Resources and further reading
----------------------------------------

- `Wikipedia <https://en.wikipedia.org/wiki/Pandas_(software)>`_



