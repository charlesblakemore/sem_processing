
Library for Processing Grayscale TIFF Images from SEM
=====================================================

A utility module and a few example scripts for processing images
captured with Scanning Electron Microscopy. The scope is currently 
quite limited, primarily with edge and circle detection, some 
filtering and binarization, and finally some calibration based on
images of 1um or 10um pitch, 2D diffraction gratings.

Install
-------

From sources
````````````

To install system-wide, noting the path to the src since no wheels
exist on PyPI, use::

   pip install ./sem_processing

If you intend to edit the code and want the import calls to reflect
those changes, install in developer mode::

   pip install -e sem_processing

If you don't want a global installation (i.e. if multiple users will
engage with and/or edit this library) and you don't want to use venv
or some equivalent::

   pip install -e sem_processing --user

where pip is pip3 for Python3 (tested on Python 3.6.9). Be careful 
NOT to use ``sudo``, as the latter two installations make a file
``easy-install.pth`` in either the global or the local directory
``lib/python3.X/site-packages/easy-install.pth``, and sudo will
mess up the permissions of this file such that uninstalling is very
complicated.


Uninstall
---------

If installed without ``sudo`` as instructed, uninstalling should be 
as easy as::

   pip uninstall sem_processing

If installed using ``sudo`` and with the ``-e`` and ``--user`` flags, 
the above uninstall will encounter an error.

Navigate to the file ``lib/python3.X/site-packages/easy-install.pth``, 
located either at  ``/usr/local/`` or ``~/.local`` and ensure there
is no entry for ``sem_processing``.


License
-------

The package is distributed under an open license (see LICENSE file for
information).

Related packages
----------------

`opt_lev_analysis <https://github.com/stanfordbeads/opt_lev_analysis>`_ - A massive
and sprawling collection of analysis and simulation scripts developed for the
Optical Levitation Project at Stanford University under the direction of Professor
Giorgio Gratta. This SEM processing library has been used to facilitate that work

Authors
-------

Charles Blakemore (chas.blakemore@gmail.com)