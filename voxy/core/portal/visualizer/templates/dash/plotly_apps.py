from dash import html
from dash.dependencies import ClientsideFunction, Input, Output
from django_plotly_dash import DjangoDash

app = DjangoDash("SampleExample")
app.scripts.append_script(
    dict(external_url="https://vjs.zencdn.net/7.18.1/video.min.js")
)
app.scripts.append_script(
    dict(external_url="/static/dash/assets/video_player.js")
)
app.layout = html.Div(
    [
        html.Canvas(
            id="canvas",
            style={"position": "absolute", "left": "20px", "top": "20px"},
        ),
        html.Video(
            id="video",
            className="video-js",
            controls=True,
            src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WhatCarCanYouGetForAGrand.mp4",
            style={"position": "absolute", "left": "20px", "top": "20px"},
        ),
        html.Div("Hello"),
    ],
    style={"position": "relative"},
)

app.clientside_callback(
    ClientsideFunction(
        namespace="clientside", function_name="large_params_function"
    ),
    Output("canvas", "children"),
    Input("canvas", "children"),
)
