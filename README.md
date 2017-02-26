# SOC

**S**earch **O**nline **C**ollections: Abstracting away datasets for data science and machine learning.

## Motivation

The goal of this project is to make it easy to create a dataset and glue it into another Python package like [Keras](https://keras.io/), without having to worry about the sticky parts. In most of datascience, getting your data into the right format is 80% of the work. The aim of PySOC is to make it grab a dataset. For example, the AskReddit module can be used to scrape data from the AskReddit subreddit through the Reddit API and store it in a serialized format, without actually having to download any data. It can then be loaded, iterated through, or otherwise manipulated using helpful abstraction methods.

## Technical Details

Each dataset is stored as a module. Data is cached in a user-specified folder, and for datasets that are scraped from public sources, like the AskReddit dataset, a command-line interface is provided for updating the cached data. The modules provide helper methods for interacting with different data types, like text.

## Installation

Pip can be used to install the package as follows:

```bash
pip install git+https://github.com/codekansas/soc
```

## Command-line Usage

The modules come with command-line tools for downloading data.

```bash
pysoc --help
```

## Example

See the Python notebook [here](/examples/ask_reddit.ipynb). This example illustrates how to build a sequence-to-sequence neural network and train it in AskReddit question-answer pairs.

## Contribute

See the [Constributing Guide](CONTRIBUTING.md).

<sub>This project was created for [Hack Illinois 2017](https://hackillinois.org/).</sub>
