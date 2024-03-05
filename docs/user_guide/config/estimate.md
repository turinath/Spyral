# Estimation Configuration

The Estimate parameters control the estimation phase. The default estimate parameters given in `config.json` are:

```json
"Estimate":
{
    "mininum_total_trajectory_points": 30,
},
```

A break down of each parameter:

## minimum_total_trajectory_points

Minimum number of points in a cluster to be considered a valid trajectory. Any clusters smaller will be ignored.
