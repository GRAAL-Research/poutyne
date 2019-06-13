# Contributing to Poutyne
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/GRAAL-Research/poutyne/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can. Try to reduce the bug to the minimum amount of code needed to reproduce: it will help in our troubleshooting procedure.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

All of the code is formatted using [yapf](https://github.com/google/yapf) with the associated [config file](https://github.com/GRAAL-Research/poutyne/blob/master/.style.yapf). In order to format the code of your submission, simply run

```
yapf poutyne --recursive --in-place
```

We also have our own `pylint` [config file](https://github.com/GRAAL-Research/poutyne/blob/master/.pylintrc). Try not to introduce code incoherences detected by the linting. You can run the linting procedure with

```
pylint poutyne
```

## Tests

If your pull request introduces a new feature, please deliver it with tests that ensure correct behavior. All of the current tests are located under the `tests` folder, if you want to see some examples. 

For any pull request submitted, **ALL** of the tests must succeed. You can run the tests with

```
pytest tests
```

## Documentation

When submitting a pull request for a new feature, try to include documentation for the new objects/modules introduced and their public methods.

 All of Poutyne's html documentation is automatically generated from the Python files' documentation. To have a preview of what the final html will look like with your modifications, first start by rebuilding the html pages.

 ```
cd docs
./rebuild_html_doc.sh
 ```

You can then see the local html files in your favorite browser. Here is an example using Firefox:

```
firefox _build/html/index.html 
```

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).
