def get_camera_uuid(video_uuid):
    return "/".join(video_uuid.split("/")[0:4])
