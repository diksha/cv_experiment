# Lambda transcode experiments

This is a lambda function and optimizer script designed to help determine whether running our
transcodes in lambda is vialbe. The input content is a 68s clip from the office camera at
720p 5fps 500kbps HEVC just like our regular streams. I've run this in two passes so far, once
with a veryfast preset (improves performance at the expense of quality) and one with ffmpeg defaults
which is basically what our current clip generators do.

## With veryfast preset

Results were as follows

Invoking with 512 MB...24.96s
Invoking with 1024 MB...12.42s
Invoking with 1536 MB...8.21s
Invoking with 2048 MB...6.15s
Invoking with 2560 MB...4.95s
Invoking with 3072 MB...4.19s
Invoking with 3584 MB...3.66s
Invoking with 4096 MB...3.15s
Invoking with 4608 MB...2.95s
Invoking with 5120 MB...2.75s
Invoking with 5632 MB...2.65s
Invoking with 6144 MB...2.33s
Invoking with 6656 MB...2.32s
Invoking with 7168 MB...2.09s
Invoking with 7680 MB...2.01s
Invoking with 8192 MB...2.02s
Invoking with 8704 MB...1.79s
Invoking with 9216 MB...1.93s
Invoking with 9728 MB...1.95s
Invoking with 10240 MB...1.88s

## With defaults

Invoking with 512 MB...68.38s
Invoking with 1024 MB...34.05s
Invoking with 1536 MB...22.52s
Invoking with 2048 MB...16.61s
Invoking with 2560 MB...13.31s
Invoking with 3072 MB...11.05s
Invoking with 3584 MB...9.73s
Invoking with 4096 MB...8.60s
Invoking with 4608 MB...7.84s
Invoking with 5120 MB...7.76s
Invoking with 5632 MB...6.34s
Invoking with 6144 MB...5.80s
Invoking with 6656 MB...5.60s
Invoking with 7168 MB...5.17s
Invoking with 7680 MB...5.28s
Invoking with 8192 MB...4.90s
Invoking with 8704 MB...4.96s
Invoking with 9216 MB...4.10s
Invoking with 9728 MB...4.40s
Invoking with 10240 MB...4.01s

## Cost estimation

Here's a spreadsheet with some cost estimates based on these results:

[Results](https://docs.google.com/spreadsheets/d/1_U3woC3qpix7I4ShEoZfwCmuIIgDpXU1GuhC1Rn48J8/edit#gid=0)
