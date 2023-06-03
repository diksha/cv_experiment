#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
from core.metaverse.metaverse import Metaverse
import argparse 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", "-u", type=str, required=True)
    return parser.parse_args()

def insert_logset(metaverse, uuid):
    result = metaverse.schema.execute(
        'mutation { logset_create(name: "lifting", videos: [ { video: "38d6cb25-238e-4c06-8dcf-9b76f643157d", violation_version: "v1", labels_version: "v1" } ]) { logset { name } }}'
    )
    print(result)

def main():
    metaverse = Metaverse(environment="INTERNAL")
    args = parse_args()
   
    insert_logset(metaverse,args.uuid)

if __name__ == "__main__":
    main()