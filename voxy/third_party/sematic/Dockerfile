#trunk-ignore-all(hadolint,semgrep)
ARG lightly_image
FROM $lightly_image
RUN apt-get -o Dpkg::Options::="--force-confmiss" install -y --no-install-recommends --reinstall netbase
RUN apt-get update && apt-get install --no-install-recommends -y libgl1-mesa-glx libglib2.0-0
RUN chmod +x /home/boris/onprem-docker/entrypoint.sh
RUN rm /home/output_dir && mkdir /home/output_dir
ENTRYPOINT "/bin/bash"
