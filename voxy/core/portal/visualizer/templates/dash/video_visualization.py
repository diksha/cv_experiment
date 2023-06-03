import json

from dash import dcc, html
from dash.dependencies import ClientsideFunction, Input, Output, State
from dash.exceptions import PreventUpdate
from django_plotly_dash import DjangoDash

from core.incidents.utils import CameraConfig
from core.infra.cloud.gcs_utils import does_gcs_blob_exists, read_from_gcs
from core.portal.lib.utils.signed_url_manager import signed_url_manager
from core.utils.video_utils import get_camera_uuid

app = DjangoDash("VideoVisualizer")
app.css.append_css(
    dict(external_url="https://vjs.zencdn.net/7.18.1/video-js.css")
)
app.scripts.append_script(
    dict(external_url="https://vjs.zencdn.net/7.18.1/video.min.js")
)
app.scripts.append_script(dict(external_url="/static/dash/assets/AVLTree.js"))
app.scripts.append_script(dict(external_url="/static/dash/assets/utils.js"))
app.scripts.append_script(
    dict(external_url="/static/dash/assets/video_player.js")
)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    ["Portal", "Voxel_video_logs", "Voxel_video_logs_h264"],
                    "Voxel_video_logs",
                    id="src-dropdown",
                ),
                dcc.Dropdown(
                    ["s3", "gcs"],
                    id="storage_src",
                    placeholder="Pick Storage Source",
                ),
                dcc.Input(
                    id="video_storage_path",
                    type="text",
                    placeholder="Relative Video Storage Path",
                    required=True,
                ),
                dcc.Input(
                    id="annotations_storage_path",
                    type="text",
                    placeholder="Relative Annotations Storage Path",
                    required=True,
                ),
                dcc.Textarea(
                    id="predictions",
                    placeholder="prediction_name1 GCSPath1\nprediction_name2 GCSPath2",
                ),
                html.Button(id="submit_button", n_clicks=0, children="Submit"),
                "Labels",
                dcc.Checklist(
                    id="pred_id_checklist",
                    style={"display": "flex", "flex-direction": "column"},
                ),
                "Categories",
                dcc.Checklist(
                    id="category_checklist",
                    style={"display": "flex", "flex-direction": "column"},
                ),
                "Attributes",
                dcc.Checklist(
                    id="attributes_checklist",
                    style={"display": "flex", "flex-direction": "column"},
                ),
                "Actors",
                dcc.Checklist(
                    id="actor_checklist",
                    style={"display": "flex", "flex-direction": "column"},
                ),
            ],
            style={
                "display": "flex",
                "flex-direction": "column",
                "width": "20%",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Video(
                            id="video",
                            className="video-js",
                            style={
                                "position": "absolute",
                                "height": "720px",
                                "width": "1280px",
                            },
                        ),
                        html.Canvas(
                            id="canvas_101",
                            style={
                                "position": "absolute",
                                "height": "720px",
                                "width": "1280px",
                            },
                        ),
                    ],
                    style={"position": "relative", "height": "720px"},
                ),
                html.Div(
                    id="attribute_information",
                    style={"position": "relative", "white-space": "pre"},
                ),
            ],
            style={
                "display": "flex",
                "flex-direction": "column",
            },
        ),
        dcc.Store(id="pred_labels"),
        dcc.Store(id="camera_config"),
        dcc.Store(id="video_loaded"),
        dcc.Store(id="empty"),
    ],
    style={
        "display": "flex",
        "flex-direction": "row",
        "margin-left": "20px",
        "margin-top": "20px",
    },
)

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="load_video"),
    Output("video_loaded", "data"),
    Input("video_loaded", "data"),
)


def getCategoryActorAttributes(allLabels):
    actors = []
    attributes = set()
    for _, value in allLabels.items():
        actors.extend(
            [actor for frame in value["frames"] for actor in frame["actors"]]
        )
    category_track_id_list = [(a["category"], a["track_id"]) for a in actors]
    attributes = {key for a in actors for key in a.keys()}
    return (
        list({i[0] for i in category_track_id_list}),
        list({i[1] for i in category_track_id_list}),
        list(attributes),
    )


def get_video_uuid(storage_path: str) -> str:
    """Get video UUID helper method

    Args:
        storage_path (str): storage path of storage URI inputted

    Returns:
        str: video uuid extrapolated from string
    """
    sub_paths = storage_path.split("/")

    for sub_path in sub_paths:
        if "mp4" in sub_path or "json" in sub_path:
            # grab first part of {UUID}_video.mp4
            return sub_path.split("_")[0]
    return ""


@app.callback(
    Output(component_id="video", component_property="src"),
    Output("pred_labels", "data"),
    Output(component_id="pred_id_checklist", component_property="options"),
    Output("category_checklist", "options"),
    Output("actor_checklist", "options"),
    Output("attributes_checklist", "options"),
    Output("camera_config", "data"),
    [Input(component_id="submit_button", component_property="n_clicks")],
    [
        State(
            component_id="annotation_storage_path", component_property="value"
        )
    ],
    [State(component_id="video_storage_path", component_property="value")],
    [State(component_id="storage_src", component_property="value")],
    [State(component_id="predictions", component_property="value")],
    [State(component_id="src-dropdown", component_property="value")],
)
def update_video_src(
    n_clicks,
    annotation_storage_path,
    video_storage_path,
    storage_src,
    predictions,
    src,
):
    """Method that grabs appropriate video from source

    Args:
        n_clicks (str): num of clicks
        annotation_storage_path (str): relative path of annotations
        video_storage_path (str): relative path of video
        storage_src (str): storage src either S3 or GCS
        predictions (str): predictions
        src (str): src of video

    Raises:
        PreventUpdate: video uuid needs to be provided
        RuntimeError: src should be portal or video logs

    Returns:
        Tuple: signed url of video path
    """
    if (
        not annotation_storage_path
        or not video_storage_path
        or not storage_src
    ):
        raise PreventUpdate

    video_uuid = get_video_uuid(video_storage_path)

    video_path = f"{storage_src}://{video_storage_path}"
    gt_path = f"{storage_src}://{annotation_storage_path}"

    if src == "Portal":
        gt_key = "Portal annotations"
    elif src == "Voxel_video_logs":
        gt_key = "Ground truth"
    elif src == "Voxel_video_logs_h264":
        gt_key = "Ground truth"
    else:
        raise RuntimeError("Src should be portal or video logs or h264")
    preds = []
    if predictions:
        preds = predictions.splitlines()
    allLabels = {}
    if does_gcs_blob_exists(gt_path):
        allLabels[gt_key] = json.loads(read_from_gcs(gt_path))
    for prediction in preds:
        [key, gcs_path] = prediction.split(" ")
        allLabels[key] = json.loads(read_from_gcs(gcs_path))
    (categories, actors, attributes) = getCategoryActorAttributes(allLabels)
    camera_config = (
        CameraConfig(get_camera_uuid(video_uuid), 1, 1).to_dict()
        if src in ("Voxel_video_logs", "Voxel_video_logs_h264")
        else None
    )
    return (
        signed_url_manager.get_signed_url(s3_path=video_path),
        list(allLabels.items()),
        list(allLabels.keys()),
        categories,
        actors,
        attributes,
        camera_config,
    )


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="draw_labels"),
    Output("empty", "data"),
    Input(component_id="video", component_property="src"),
    Input("pred_labels", "data"),
    Input("pred_id_checklist", "value"),
    Input("category_checklist", "value"),
    Input("actor_checklist", "value"),
    Input("attributes_checklist", "value"),
    Input("camera_config", "data"),
)
