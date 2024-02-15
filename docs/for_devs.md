# For Developers

Spyral is always looking for input and improvements! Pull requests are always welcome, however, they may not always be accepted. Spyral aims to be as experiment agnostic as possible; this means that some improvements which only work for one type of experiment may not be reasonable to integrate with the framework. Please keep this in mind when submitting a new feature!

Below we've outlined some of the rules for additions to the Spyral framework:

## Statement of Intent

Please state in the pull request what the goal of is of this modification to Spyral and what was changed to accomplish this. If there is no statement of intent, the pull request will not be accepted.

## Statement of Dependencies

Please state in the pull request what new dependencies (if any) were added and please make sure that these new dependencies were added to the Spyral `requirements.txt` with a version pinned. If the dependencies were not stated, the pull request will not be accepted. Please check to make sure that any new dependencies are compliant with the [GPL3](https://www.gnu.org/licenses/gpl-3.0.en.html) license which is used by Spyral.

## Code Requirements

Below are some of the requirements for any code added to Spyral:

### Docstrings

Please provide docstrings for each function and class. Dataclasses do not necessarily need docstrings. This helps other developers understand what the purpose and usage code. Check out some of the docstrings in Spyral to get a feel for the expected format.

### Type Hints

Please use [type hints](https://docs.python.org/3/library/typing.html) to annotate your code where applicable. This includes function arguments, return values, and any variables where the type might be ambiguous. In general, a good rule to follow is: can your IDE determine the type of the variable? If it can it doesn't need a type hint. If it can't, the variable needs a type hint. The Any type hint is allowed in some specific cases. Functions which return None do not need a return type hint.

### Formatting

TBD

### Files

Please try to keep files from being monster 1000 lines of code documents. This is not a hard and fast rule, but in general files should contain a unqiue individual unit of code which interfaces with the rest of the framework. There can be execptions.

### Notebooks

The main rule for contributing notebooks is that they should only really show example implementations of Spyral in action, not provide any unique analysis that isn't included in the parent Spyral framework. That is, you shouldn't be implementing new functions or classes in the notebooks; you should only be using the tools already available in Spyral or its dependencies.

### Documentation

If you decide to contribute a new major feature to Spyral, please prepeare some documentation to be added to this site. It should at least outline what new configuration parameters are exposed and what effect this may have upon the data. Documentation should be contributed in the form of Markdown files in the `docs` directory. Our documenation is built using the amazing [MkDocs](https://www.mkdocs.org/) and the [MkDocs-Material](https://squidfunk.github.io/mkdocs-material/) theme.

## Final Thoughts

Below is an example pull request descripton:

```txt
Intent:
Fix a bug in configuration parsing in Spyral and switch to YAML

Dependencies Added:
pyyaml
```

Feel free to use this as a simple template if you wish!

If you find that you need to really customize your Spyral to fit your specific experiment, please consider forking the parent Spyral repository. This will allow you to still get all the power of GitHub and version control, without having to make your code necessarily complaint with Spyral's experiment agnostic goals.
