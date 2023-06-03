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

import flask
import json
import argparse
import os


app = flask.Flask(__name__)

image_path = None
focal_length = None

@app.route('/')
@app.route('/index.html')
def root():
    global image_path
    global focal_length
    if image_path is not None and focal_length is not None:
        return flask.render_template('extrinsics.html', image="default", focal_length=focal_length)
    else:
        return "Please pass a image path and a focal length, or navigate to \n \
                localhost:8080/image/image_name/focal_length.  Where \"image_name\" is a image in the static \n \
                directory and the \"focal length\" is generated from deep calib. Like: localhost:8080/image/warehouse.jpg/299"

@app.route('/load/<path>')
def send_image(path):
    # this adds support for image path
    if path == "default" and image_path is not None:
        return flask.send_file(image_path)
    return flask.send_from_directory('static', path)

@app.route('/image/<path>/<focal_length>')
def dynamic_image(path, focal_length):
    return flask.render_template('extrinsics.html', image=path, focal_length=focal_length)

@app.route('/save/<image>/<pitch>/<roll>/<height>')
def save_config(image, pitch, roll, height):
    global image_path
    global focal_length
    config = {"pitch_rad": pitch , "roll_rad": roll, "height_m": height }
    image = image.replace(".", "_")
    home = os.path.expanduser("~")
    filename = f"{home}/{image}_extrinsics.json"
    if image_path is not None and focal_length is not None:
        filename = image_path.replace("/", "_").replace("~", "").replace(".", "_") + "_extrinsics.json"

    json_config = json.dumps(config, indent=4)
    with open(filename, "w" ) as f:
        print("writing config out to " + filename)
        f.write(json_config)
    return json_config



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate calibration from an image and focal length')
    parser.add_argument('--focal_length', type=float, default=299.0,
			help='the focal length in image coordinates based on 299x299 sized image')
    parser.add_argument('--image_path', type=str,
			help='the image path')
    args = parser.parse_args()
    image_path = args.image_path
    focal_length     = args.focal_length
    app.run(host='127.0.0.1', port=8080, debug=True)
