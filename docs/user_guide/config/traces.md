# GET & FRIB DAQ Trace Analysis Configuration

The GET and FRIB parameters control the trace analysis for each acquisition. The default detector parameters in `config.json` are:

```json
"GET":
{
    "baseline_window_scale": 20.0,
    "peak_separation": 50.0,
    "peak_prominence": 20.0,
    "peak_max_width": 100.0,
    "peak_threshold": 25.0
},

"FRIB":
{
    "baseline_window_scale": 100.0,
    "peak_separation": 50.0,
    "peak_prominence": 20.0,
    "peak_max_width": 500.0,
    "peak_threshold": 100.0,
    "event_ic_multiplicity": 1,
    "event_correct_ic_time": true
},
```

A break down of each parameter:

## baseline_window_scale

The size of the window used to create a moving average of the baseline in the baseline removal algorithm. Baselines are removed using fourier analysis, passing the traces through a low-pass filter. This parameter is available to both FRIB and GET data.

## peak_separation

The minimum space between peaks in the peak finding algorithm. This is a re-exposure of the `distance` parameter in [scipy's find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks). See their documetation for more details. This parameter is available to both FRIB and GET data.

## peak_prominence

The minimum prominence of a peak (distance to the lowest contour). This is a re-exposure of the `prominence` parameter in [scipy's find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks). See their documentation for more details. This parameter is available to both FRIB and GET data.

## peak_max_width

The maximum width of a peak (at the base of the peak). This is a re-exposure of the `width` parameter in [scipy's find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks). See their documentation for more details. This parameter is available to both FRIB and GET data.

## peak_threshold

The minimum amplitude of a peak. This parameter is available to both FRIB and GET data.

## event_ic_multiplicity

The maximum allowed ion chamber multiplicity for an event. In general, AT-TPC experiments would only allow for one ion chamber hit per event. This is only available to FRIB data.

## event_correct_ic_time

Turn on/off the ion chamber time correction. Sometimes, the wrong beam triggers the event start in the AT-TPC. This can be corrected using the correlation between downstream silicon and the ion chamber. `true` turns the correction on, `false` turns the correction off. This is only available to FRIB data.

## Important Note

If the `is_legacy` switch is turned on in the Run parameters, the FRIB parameters are used to process the ion chamber, even though the ion chamber data is recorded through the GET acquisition. This allows for the setting of independent parameters for analyzing the IC. However some of the FRIB parameters are unused in this case. In particular, the `event_correct_ic_time` and `event_ic_multiplicity` are not used. Additionally, the baseline of the IC signal is analyzed using the GET `baseline_window_scale`.
