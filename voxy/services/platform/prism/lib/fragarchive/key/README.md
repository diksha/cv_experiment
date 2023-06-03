# Fragment Key Format

This document describes the format of keys for video fragments stored in S3

A fragment key is the string concatenation of the source `CameraUUID`, a `/` delimiter, and a 10-digit UNIX `timestamp` (seconds since Unix epoch), and a filetype extension (currently only `.mkv`)

Example: `wesco/reno/0002/cha/1679424920.mkv`

## Timestamp

The timestamp is the producer's timestamp of the video fragment and represents the 'start time' of the fragment.

For sorting purposes, the timestamp must be exactly 10 digits long.

Zero-padding is not supported in order to reduce the chance of incorrect timestamps.

### Extreme Values

The earliest valid timestamp is 1000000000 which represents 'Sep 09 2001 01:46:40 GMT+0000'.

The latest valid timestamp is 9999999999 which represents 'Nov 20 2286 17:46:39 GMT+0000'

## Example

CameraUUID: wesco/reno/0002/cha

Timestamp: Tue Mar 21 2023 20:26:37 GMT+0000 = 1679430397 seconds since Unix Epoch

File type: MKV

fragkey = wesco/reno/0002/cha/1679430397.mkv
