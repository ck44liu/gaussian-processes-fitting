# gaussian-processes-fitting
This project uses Gaussian Processes to investigate the occurance of muscle co-contraction when an individual is tracing a curve. Both global window approach and sliding window approach are utilized. We use radial-basis function for the kernel. The co-contraction occurs when we observe a sudden change, namely a spike, of `sigma^2` and `length_scale` across different local windows.
## Files
- `gp_mk_global.py` and `gp_mk_slidewindow.py` are global window approach (fit one Gaussian) and sliding window approach (fit multiple small scale Gaussians across the timeframe) respectively.
- `tracing data` contains the data from the curve-tracing individual. There are five trials in total. We use the first four trials for traning and the last trial for prediction. It also includes an explanation of the data regarding the tracking points and the meaning of different features.
