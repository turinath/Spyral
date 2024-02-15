# Frequently Asked Questions

## Where can I learn about a specific function/class/code in Spyral?

See the [API docs](api/index.md) for API level documentation.

Spyral has docstrings and comments extensively in the code. If you are writing or using Spyral and want to know what is happening in a specific part, the documentation on this site can give you an overview, but the real fine-grained details will be self documented in the code.

## How can I contriubte to Spyral?

See [For Developers](for_devs.md)

## I just downloaded Spyral and nothing is working

See the [User Guide](user_guide/getting_started.md)

## What do all these configuration parameters mean?

See [Configuration](user_guide/config/about.md)

## I have some old data I want to analyze, can I use Spyral?

In general, yes. Spyral has a legacy mode that supports analyzing data taken before AT-TPC's data acquisition was split (pre-FRIBDAQ): see [here](user_guide/phases/point_cloud.md) for more details. However, at this time, it only supports analyzing the ion chamber data out of CoBo 10. Any other auxilary signals will have to be extended or requested.

## Why can't I install Spyral with pip (i.e. `pip install spyral`)? Why can't I `import spyral`?

Spyral is not a library/package. Spyral is an application, intended to be run by users through the provided entry points (notebooks or scripts). This is also reflected in the way Spyral handles dependencies. We pin versions explicitly, to ensure that the code will run identically in whatever environment it is installed to.

More importantly, it will be extremely rare that Spyral works perfectly right out of the box for any analysis; the AT-TPC is such a powerful tool, that can be used to take such a wide range of data, that no one single analysis will work perfectly for all of these use cases. A good example of this is the so-called estimation phase of AT-TPC analysis where we attempt to guess the physical properties of a trajectory for use in the final fitting/solving phase. This phase is extremely sensitive to the shape/size of the data being analyzed. As such, even more so than an application, Spyral is a *framework*, a ground floor providing the basic building blocks of AT-TPC analysis. We want users to have access to the source code, and be able to modify the behavior to fit their needs without having to be restricted by our design choices, because we cannot predict every use case.

This may change in the future, in particular as Spyral matures and becomes more tested, it may become clear that a more rigid analysis environment will provide users with a better access point to AT-TPC analysis. In such as case, Spyral may migrate to an installable package.

Please let us know what you think on this topic! User feedback will be critical for determining the path forward for Spyral.
