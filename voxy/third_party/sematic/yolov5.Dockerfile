#trunk-ignore-all(hadolint,semgrep)
FROM 203670452561.dkr.ecr.us-west-2.amazonaws.com/sematic:voxel-yolov5-2023-03-29T20-31-25
RUN apt-get -o Dpkg::Options::="--force-confmiss" install -y --no-install-recommends --reinstall netbase
RUN apt-get update && apt-get install --no-install-recommends -y libgl1-mesa-glx libglib2.0-0
ENTRYPOINT "/bin/bash"