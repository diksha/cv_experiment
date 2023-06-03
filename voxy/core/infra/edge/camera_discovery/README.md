# Camera discovery

## Capture video clips

Use this script to capture short video clips from a list of cameras. This is useful when identifying which cameras we want to run Voxel systems on. It's probably easiest to just copy the content of `capture_clips.sh` into a new file on the remote host.

### Step 1 - Create camera list file

Create a `cameras.txt` file containing a list of authorized RTSP URIs (one per line). URIs can be a mix of cameras and NVRs, as long as they are valid RTSP URIs. It should look something like this:

```text
rtsp://admin:admin@11.12.13.14:554
rtsp://foo:bar@nvr1.example.com:554/trackID=1
rtsp://foo:bar@nvr2.example.com:554/trackID=1&streamID=2
```

### Step 2 - Run the script

Run the `capture_clips.sh` script:

- (-c) camera list file
- (-o) capture output directory
- (-t) ffmpeg timeout (e.g. approximate clip length)

```shell
./capture_clips.sh -c cameras.txt -o captures -t 10
```

Depending on how many cameras you're capturing from and how long your specified timeout is, this could take a while...

### Step 3 - Review the results

Video files will be available in the output directory you specified. One easy way to move these files to your local machine for easier review is to use a tool called [`magic-wormhole`](https://magic-wormhole.readthedocs.io/en/latest/).

#### Install on the remote host

```bash
sudo apt install magic-wormhole
```

#### Install on your local machine (assuming Mac, see docs for more options)

```bash
brew install magic-wormhole
```

#### Open a wormhole to the output directory on the remote host

```bash
wormhole send captures

Sending 7924 byte file named 'captures'
On the other computer, please run: wormhole receive
Wormhole code is: 7-crossover-clockwork

Sending (<-10.0.1.43:58988)..
100%|=========================| 7.92K/7.92K [00:00<00:00, 6.02MB/s]
File sent.. waiting for confirmation
Confirmation received. Transfer complete.
```

This will print a wormhole code, you'll need this for the next step.

#### Receive the wormhole data on your local machine

```bash
wormhole receive

Enter receive wormhole code: 7-crossover-clockwork
Receiving file (7924 bytes) into: captures
ok? (y/n): y
Receiving (->tcp:10.0.1.43:58986)..
100%|===========================| 7.92K/7.92K [00:00<00:00, 120KB/s]
Received file written to captures
```

That's it! `wormhole` will drop the directory into your current working directory.
