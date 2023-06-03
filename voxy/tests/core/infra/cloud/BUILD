load("@pip_deps//:requirements.bzl", "requirement")
load("//third_party/pytest:build_defs.bzl", "voxel_py_test")

voxel_py_test(
    name = "kinesis_utils_test",
    srcs = [
        "kinesis_utils_test.py",
    ],
    data = [
        "@artifacts_office_cam_mp4",
    ],
    tags = ["requires-network"],
    deps = [
        requirement("moto"),
        requirement("boto3"),
        requirement("av"),
        "//core/infra/cloud:kinesis_utils",
        "//core/utils:pyav_decoder",
    ],
)

voxel_py_test(
    name = "utils_test",
    srcs = [
        "utils_test.py",
    ],
    deps = [
        "//core/infra/cloud:utils",
    ],
)

voxel_py_test(
    name = "gcs_utils_test",
    srcs = [
        "gcs_utils_test.py",
    ],
    tags = ["requires-network"],
    deps = [
        "//core/infra/cloud:gcs_utils",
    ],
)
