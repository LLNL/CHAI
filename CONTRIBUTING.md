# Contributing to CHAI

This document is intented for developers who want to add new features or
bugfixes to CHAI. It assumes you have some familiarity with git and GitHub. It
will discuss what a good pull request (PR) looks like, and the tests that your
PR must pass before it can be merged into CHAI.

## Forking CHAI

If you aren't a CHAI deveolper at LLNL, then you won't have permission to push
new branches to the repository. First, you should create a
[fork](https://github.com/LLNL/CHAI#fork-destination-box). This will create a
copy of the CHAI repository that you own, and will ensure you can push your
changes up to GitHub and create pull requests.

## Developing a New Feature

New features should be based on the `develop` branch. When you want to create a
new feature, first ensure you have an up-to-date copy of the `develop` branch:

    $ git checkout develop
    $ git pull origin develop

You can now create a new branch to develop your feature on:

    $ git checkout -b feature/<name-of-feature>

Proceed to develop your feature on this branch, and add tests that will exercise
your new code. If you are creating new methods or classes, please add Doxygen
documentation.

Once your feature is complete and your tests are passing, you can push your
branch to GitHub and create a PR.

## Developing a Bug Fix

First, check if the change you want to make has been fixed in `develop`. If so,
we suggest you either start using the `develop` branch, or temporarily apply the
fix to whichever version of CHAI you are using.

Assuming there is an unsolved bug, first make sure you have an up-to-date copy
of the develop branch:

    $ git checkout develop
    $ git pull origin develop

Then create a new branch for your bugfix:

    $ git checkout -b bugfix/<name-of-bug>

First, add a test that reproduces the bug you have found. Then develop your
bugfix as normal, and ensure to `make test` to check your changes actually fix
the bug.

Once you are finished, you can push your branch to GitHub, then create a PR.

## Creating a Pull Request

You can create a new PR [here](https://github.com/LLNL/CHAI/compare). GitHub
has a good [guide](https://help.github.com/articles/about-pull-requests/) to PR
basics if you want some more information. Ensure that your PR base is the
`develop` branch of CHAI.

Add a descriptive title explaining the bug you fixed or the feature you have
added, and put a longer description of the changes you have made in the comment
box.

Once your PR has been created, it will be run through our automated tests and
also be reviewed by RAJA team members. Providing the branch passes both the
tests and reviews, it will be merged into RAJA.

## Tests

CHAI's tests are all in the `test` directory, and your PR must pass all these tests
before it is merged. If you are adding a new feature, add new tests.
