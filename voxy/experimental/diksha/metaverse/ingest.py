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
import csv
import os

from core.metaverse.metaverse import Metaverse

def main():
    metaverse = Metaverse(environment="INTERNAL")
    ws = os.path.dirname(__file__)
    with open(os.path.join(ws, "CameraMetadata.csv"), newline="") as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            query_string = "mutation { camera_create("
            remove_braces = str(row)[1:-1]
            row_values = remove_braces.split(',')
            for row_value in row_values:
                key_val = row_value.strip().split(':')
                key = key_val[0].strip()[1:-1]
                value = "\"" + key_val[1].strip()[1:-1] + "\""
                if value.lower() == "\"true\"" or value.lower() == "\"false\"":
                    value = value.lower().strip()[1:-1]
                query_string += (key + ": " + value + ", ")
            query_string = query_string[0:-2]
            query_string += ") { camera { uuid } }}"
            result = metaverse.schema.execute(query_string)
            print(result)
    
if __name__ == "__main__":
    main()