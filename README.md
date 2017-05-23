Python tools for the filtering of digital signals.


Installation:
=============

... on GA's Iris cluster:
-------------------------
Package management is cleanly handled on Iris via
[modules](https://diii-d.gat.com/diii-d/Iris#Environment_modules).
The `filters` package has a corresponding modulefile
[here](https://github.com/emd/modulefiles).

To use the `filters` package, change to the directory
you'd like to download the source files to and
retrieve the source files from github by typing

    $ git clone https://github.com/emd/filters.git

The created `filters` directory defines the
package's top-level directory.
The modulefiles should be similarly cloned.

Now, at the top of the corresponding
[modulefile](https://github.com/emd/modulefiles/blob/master/filters),
there is a TCL variable named `filters_root`;
this must be altered to point at the
top-level directory of the cloned `filters` package.
That's it! You shouldn't need to change anything else in
the modulefile. The `filters` module can
then be loaded, unloaded, etc., as is discussed in the
above-linked Iris documentation.

The modulefile also defines a series of automated tests
for the `filters` package. Run these tests at the command line
by typing

    $ test_filters

If the tests return "OK", the installation should be working.

... elsewhere:
--------------
Change to the directory you'd like to download the source files to
and retrieve the source files from github by typing

    $ git clone https://github.com/emd/filters.git

Change into the `filters` top-level directory by typing

    $ cd filters

For accounts with root access, install by running

    $ python setup.py install

For accounts without root access (e.g. a standard account on GA's Venus
cluster), install locally by running

    $ python setup.py install --user

To test your installation, run

    $ nosetests tests/

If the tests return "OK", the installation should be working.


Use:
====
