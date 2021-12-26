---
title: Installation and Getting Started
---

# Installation

The first step for working with Python is installing Python itself. The easiest way is to install Python via command line tools.

> ## Command Line Notation
>
> In this chapter and throughout the book, we’ll show some commands used in the
> terminal. Lines that you should enter in a terminal all start with `$`. You
> don’t need to type in the `$` character; it indicates the start of each
> command. Lines that don’t start with `$` typically show the output of the
> previous command. Additionally, PowerShell-specific examples will use `>`
> rather than `$`.

# Installing Python on macOS

If you are using macOS and you have `brew` installed, then you can enter the following command:

```console
$ brew install python
```

# Conventions Used

In this book, we will focus on the following libraries and their APIs:

*  `scikit-learn`: for building machine learning models and pipelines

For introducing ideas around transfer learning, we will use the following libraries - and their pre-built models:

*  `transformers`: the huggingface library
*  `keras`: the deep learning library

# Who Productionises Machine Learning?

In small organisations, the person who would build a model would productionise the model! Based on this you might wonder "how would we scale this"? By keeping machine learning systems simple, it becomes _easier_ to scale and manage. 

Of course in smaller organisations, you also wouldn't necessarily have "big data" challenges, your data would only be moderately sized! 

Nevertheless, we will touch on these topics within this book.


