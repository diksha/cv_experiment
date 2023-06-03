"""This module loads voxel artifacts from the voxel-storage s3 bucket"""

load("//rules:s3_archive.bzl", "s3_archive")

def voxel_artifact(**kwargs):
    """Loads a single voxel artifact"""

    if "build_file_content" in kwargs:
        fail("build_file_content not supported for voxel_artifact")

    s3_archive(
        build_file_content = """
genrule(
    name = "meta",
    outs = ["meta.json"],
    cmd = \"\"\"cat <<EOF>$@
{}
EOF\"\"\",
)

filegroup(
    name = "{}",
    srcs = ["meta.json"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "preload",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
""".format(json.encode(kwargs), kwargs["name"]),
        **kwargs
    )

def voxel_artifacts():
    """Loads voxel artifacts"""

    voxel_artifact(
        name = "artifacts_doors_0630",
        sha256 = "03b3b0b53ade9e1c12a38e77b17cb07fd097e4c90280b309f91e478f0e7ed9bd",
        url = "s3://voxel-storage/artifactory/doors_0630/03b3b0b53ade9e1c12a38e77b17cb07fd097e4c90280b309f91e478f0e7ed9bd/doors_0630.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_door_classifier_20210602123803",
        sha256 = "47aa082b01245776c05d3113c0674df89aca6c3525ced2d621e01a97b6a93b8a",
        url = "s3://voxel-storage/artifactory/door_classifier_20210602123803/47aa082b01245776c05d3113c0674df89aca6c3525ced2d621e01a97b6a93b8a/door_classifier_20210602123803.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_door-classifier-bloomingdale-0006-05-03",
        sha256 = "5d6d20f7e289441f5acfbfc99e1e2d911bd661ebbc58d66edc17326084f052d7",
        url = "s3://voxel-storage/artifactory/door-classifier-bloomingdale-0006-05-03/5d6d20f7e289441f5acfbfc99e1e2d911bd661ebbc58d66edc17326084f052d7/door-classifier-bloomingdale-0006-05-03.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_door-classifier-bloomingdale-0001-05-02",
        sha256 = "5260f575454d793b461b6deb22f1a0ef61bf814980b2c72e043ee23526766a57",
        url = "s3://voxel-storage/artifactory/door-classifier-bloomingdale-0001-05-02/5260f575454d793b461b6deb22f1a0ef61bf814980b2c72e043ee23526766a57/door-classifier-bloomingdale-0001-05-02.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_pose_0630",
        sha256 = "046fb905c58eaab42007734710561537ef1196b50ccc333fa9acae7927fa9572",
        url = "s3://voxel-storage/artifactory/pose_0630/046fb905c58eaab42007734710561537ef1196b50ccc333fa9acae7927fa9572/pose_0630.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_yolo-person-pit-exp2-dataset3",
        sha256 = "90468ef0efe0d289e20d3f9bd8bb392791832ee7b4efe40f431d6eacd7bb7395",
        url = "s3://voxel-storage/artifactory/yolo-person-pit-exp2-dataset3/90468ef0efe0d289e20d3f9bd8bb392791832ee7b4efe40f431d6eacd7bb7395/yolo-person-pit-exp2-dataset3.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_door_classifier_resnet50_dataset4",
        sha256 = "269a2c7fff08aa2cb6cbb3e3c56a0f7282a843f232104d7e4560935c46f610b1",
        url = "s3://voxel-storage/artifactory/door_classifier_resnet50_dataset4/269a2c7fff08aa2cb6cbb3e3c56a0f7282a843f232104d7e4560935c46f610b1/door_classifier_resnet50_dataset4.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_americold_model",
        sha256 = "9567ad49d1eeb9e212a18268c9420e199e521bcfae8ccdbda9efac72eb4cc0fd",
        url = "s3://voxel-storage/artifactory/americold_model/9567ad49d1eeb9e212a18268c9420e199e521bcfae8ccdbda9efac72eb4cc0fd/americold_model.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_pose_classifier_90",
        sha256 = "a7daf5f602737f05dc7af02a3faad15afc44e0f8aa1e54fc0fc855b6e6095709",
        url = "s3://voxel-storage/artifactory/pose_classifier_90/a7daf5f602737f05dc7af02a3faad15afc44e0f8aa1e54fc0fc855b6e6095709/pose_classifier_90.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_aws_iot_securetunneling_localproxy",
        sha256 = "b373528b53427c4a817807c893a9de0160adb4e729c5f41670569a676d298ea6",
        url = "s3://voxel-storage/artifactory/aws_iot_securetunneling_localproxy/b373528b53427c4a817807c893a9de0160adb4e729c5f41670569a676d298ea6/aws_iot_securetunneling_localproxy.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_vest_classifier-9-19",
        sha256 = "85d52f8dac98131a5cd0f24de9945274be3f30e7a4593d8c275e91a1e3519d20",
        url = "s3://voxel-storage/artifactory/vest_classifier-9-19/85d52f8dac98131a5cd0f24de9945274be3f30e7a4593d8c275e91a1e3519d20/vest_classifier-9-19.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_vest_classifier-10-26",
        sha256 = "6dff0293bc788c0c32c6855c1f010191b8797a2fc376b08cffc1eb9ed28f38e9",
        url = "s3://voxel-storage/artifactory/vest_classifier-10-26/6dff0293bc788c0c32c6855c1f010191b8797a2fc376b08cffc1eb9ed28f38e9/vest_classifier-10-26.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_reach_classifier_01242022",
        sha256 = "14009b43e76dabe4ef9c8ffa3941dff0aeff3b5d8f7b56d9c9bd7f1802fa6a9b",
        url = "s3://voxel-storage/artifactory/reach_classifier_01242022/14009b43e76dabe4ef9c8ffa3941dff0aeff3b5d8f7b56d9c9bd7f1802fa6a9b/reach_classifier_01242022.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_lift_classifier_01192022",
        sha256 = "09e8a4f5aa9b6f27e9279b86ab7e8614897900b5781f30729303d4bf290c75f1",
        url = "s3://voxel-storage/artifactory/lift_classifier_01192022/09e8a4f5aa9b6f27e9279b86ab7e8614897900b5781f30729303d4bf290c75f1/lift_classifier_01192022.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_lift_classifier_01282022",
        sha256 = "475f12e5bec3ffaef1a9298c11026ad59b6d5681cefdb35d99c092445f01ce1f",
        url = "s3://voxel-storage/artifactory/lift_classifier_01282022/475f12e5bec3ffaef1a9298c11026ad59b6d5681cefdb35d99c092445f01ce1f/lift_classifier_01282022.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_edge-production-temp-key",
        sha256 = "abc2328f6102bd7ed97652eb129cbebc6584b08c9a9e17dbe498af8471bbc9d1",
        url = "s3://voxel-storage/artifactory/edge-production-temp-key/abc2328f6102bd7ed97652eb129cbebc6584b08c9a9e17dbe498af8471bbc9d1/edge-production-temp-key.tar.gz",
    )

    # Created by Vai. Resample the Americold door data
    voxel_artifact(
        name = "artifacts_door_classifier_resnet50_20211129",
        sha256 = "ee61cc52888a63658dad714de621ea18a69ae4f8bbc11310032c9108fd467bca",
        url = "s3://voxel-storage/artifactory/door_classifier_resnet50_20211129/ee61cc52888a63658dad714de621ea18a69ae4f8bbc11310032c9108fd467bca/door_classifier_resnet50_20211129.tar.gz",
    )

    # Experimental door model with model training framework
    voxel_artifact(
        name = "artifacts_628cf2c8-4677-47e6-9af4-4bc1b098a304",
        sha256 = "ee7cc0f85fad63263f30922007723689f98586bf3904f0cd2e1fafd0de2fe54f",
        url = "s3://voxel-storage/artifactory/628cf2c8-4677-47e6-9af4-4bc1b098a304/ee7cc0f85fad63263f30922007723689f98586bf3904f0cd2e1fafd0de2fe54f/628cf2c8-4677-47e6-9af4-4bc1b098a304.tar.gz",
    )

    #################################################################################################

    #################################################################################################

    # Vai: Door Classifier trained on dark data
    voxel_artifact(
        name = "artifacts_door_classifier_resnet50_2021-12-27",
        sha256 = "34aa20bebd4e57d9ab1362b64082df4650ef984c82e727cfd7c64b84f525c550",
        url = "s3://voxel-storage/artifactory/door_classifier_resnet50_2021-12-27/34aa20bebd4e57d9ab1362b64082df4650ef984c82e727cfd7c64b84f525c550/door_classifier_resnet50_2021-12-27.tar.gz",
    )

    # Vai: Hat classifier
    voxel_artifact(
        name = "artifacts_hat_classifier_01192022",
        sha256 = "227a9cbbdfd5dc1c2071b00879e6bd9e9147f815efad78323592804fc085b028",
        url = "s3://voxel-storage/artifactory/hat_classifier_01192022/227a9cbbdfd5dc1c2071b00879e6bd9e9147f815efad78323592804fc085b028/hat_classifier_01192022.tar.gz",
    )

    # Mamrez: Model for Safety Vest With attention
    voxel_artifact(
        name = "artifacts_attention_vestclassifier",
        sha256 = "3fe135a7069e98beafb59bfad2da43ec228c3f7a836a8924c17205f8a9921279",
        url = "s3://voxel-storage/artifactory/attention_vestclassifier/3fe135a7069e98beafb59bfad2da43ec228c3f7a836a8924c17205f8a9921279/attention_vestclassifier.tar.gz",
    )

    #Tim: the deepcalib pretrained weights for single net
    voxel_artifact(
        name = "artifacts_deepcalib_pretrained_weights",
        sha256 = "6bb1e430c65a22da794f899ee5fcc51ff598f74fdcdc976acbca37db4e65d088",
        url = "s3://voxel-storage/artifactory/deepcalib_pretrained_weights/6bb1e430c65a22da794f899ee5fcc51ff598f74fdcdc976acbca37db4e65d088/deepcalib_pretrained_weights.tar.gz",
    )

    # Vai: Best performing Vest classifier with synthetic data 04/01/22
    voxel_artifact(
        name = "artifacts_voxel_safety_vest_synthetic_classifier_attention_resnet_focal_loss_gamma1.5_WS_2022-04-01",
        sha256 = "7413d6ac6abb28cb928cf09197672216fb2d0a42bd7a4f5efc93068ea54cde74",
        url = "s3://voxel-storage/artifactory/voxel_safety_vest_synthetic_classifier_attention_resnet_focal_loss_gamma1.5_WS_2022-04-01/7413d6ac6abb28cb928cf09197672216fb2d0a42bd7a4f5efc93068ea54cde74/voxel_safety_vest_synthetic_classifier_attention_resnet_focal_loss_gamma1.5_WS_2022-04-01.tar.gz",
    )

    # Mamrez: Best performing Vest classifier with vit model 04/19/22
    voxel_artifact(
        name = "artifacts_vit_vest",
        sha256 = "973c2737fc36ea224f18d6af180b2a203aa8c2f65942f93b30177c0ac974a6c9",
        url = "s3://voxel-storage/artifactory/vit_vest/973c2737fc36ea224f18d6af180b2a203aa8c2f65942f93b30177c0ac974a6c9/vit_vest.tar.gz",
    )

    # Gabriel: Generalized YOLOv5 model for ontario deployment 05/06/22
    voxel_artifact(
        name = "artifacts_yolo_20220502_generic_v1",
        sha256 = "af7ec260bdacf31339831f922324020ac35335f27f67c20750c4af2f4eacfc78",
        url = "s3://voxel-storage/artifactory/yolo_20220502_generic_v1/af7ec260bdacf31339831f922324020ac35335f27f67c20750c4af2f4eacfc78/yolo_20220502_generic_v1.tar.gz",
    )

    # Mamrez: Ergonomic ML model trained on Synthetic data 05/11/2022
    voxel_artifact(
        name = "artifacts_ergo_ml",
        sha256 = "6c99f1467dc9bbc75f2a43b3488ebb33f4da432d5484debf09705f27cdca1c3e",
        url = "s3://voxel-storage/artifactory/ergo_ml/6c99f1467dc9bbc75f2a43b3488ebb33f4da432d5484debf09705f27cdca1c3e/ergo_ml.tar.gz",
    )

    #Mamrez: Overreaching Ergonomic ML model trained on Synthetic data 05/23/2022
    voxel_artifact(
        name = "artifacts_voxel_ergo_ml_overreaching",
        sha256 = "5c2336be9861f4fa9dfc9c7d88b283d244f1308b435c7bb475f59a1412ab32b5",
        url = "s3://voxel-storage/artifactory/voxel_ergo_ml_overreaching/5c2336be9861f4fa9dfc9c7d88b283d244f1308b435c7bb475f59a1412ab32b5/voxel_ergo_ml_overreaching.tar.gz",
    )

    # Gabriel: BuildersFirstSource SolanaBeach Doors
    voxel_artifact(
        name = "artifacts_2022-08-26-T18-42buildersfirstsource_solanabeach_0009_cha",
        sha256 = "8a9b8ce6da10b9d6b90ef05ea4217cf9c17c99e06a1344d04655206f2a942a5b",
        url = "s3://voxel-storage/artifactory/2022-08-26-T18-42buildersfirstsource_solanabeach_0009_cha/8a9b8ce6da10b9d6b90ef05ea4217cf9c17c99e06a1344d04655206f2a942a5b/2022-08-26-T18-42buildersfirstsource_solanabeach_0009_cha.tar.gz",
    )

    # Gabriel: BuildersFirstSource SolanaBeach Vest
    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vit_2022-08-30_solana_beach",
        sha256 = "6797c111272af73226ee833bf9685e5cfc9e962a71dad2381d62f8f54fbc31e0",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_2022-08-30_solana_beach/6797c111272af73226ee833bf9685e5cfc9e962a71dad2381d62f8f54fbc31e0/voxel_safetyvest_vit_2022-08-30_solana_beach.tar.gz",
    )

    # Gabriel: Quakertown Safety Vest Classifier
    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vit_quakertown_laredo_2022-08-24",
        sha256 = "c01cfb86630388266261f3cd7be19450e154c78628e326f21d8b450ebabcf5bd",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_quakertown_laredo_2022-08-24/c01cfb86630388266261f3cd7be19450e154c78628e326f21d8b450ebabcf5bd/voxel_safetyvest_vit_quakertown_laredo_2022-08-24.tar.gz",
    )

    # Gabriel: Quakertown, Solana Beach, Dixieline YOLO Model
    voxel_artifact(
        name = "artifacts_2022-08-31-01-26-05-aa61-yolo-torchscript",
        sha256 = "6eda9677b9297ed572ce68e2523bb9c17d37b53d0e7017f0910d5857273dbe11",
        url = "s3://voxel-storage/artifactory/2022-08-31-01-26-05-aa61-yolo/6eda9677b9297ed572ce68e2523bb9c17d37b53d0e7017f0910d5857273dbe11/2022-08-31-01-26-05-aa61-yolo.tar.gz",
    )

    #Nasha: Lakeshore, Arlington Safety Vest Classifier 2022/09/12

    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vit-lakeshore-arlington_20022-09-08",
        sha256 = "f4b0b7dbfed066325145d537eafa41b814804716ba48544c1bb231f22fac99e2",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit-lakeshore-arlington_20022-09-08/f4b0b7dbfed066325145d537eafa41b814804716ba48544c1bb231f22fac99e2/voxel_safetyvest_vit-lakeshore-arlington_20022-09-08.tar.gz",
    )

    #Nasha: USCOLD, Laredo Safety Vest Classifier 2022/09/15

    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vit-uscold-laredo_2022-09-15",
        sha256 = "323be8174bcd833a742254ae6f2310a3373007b95de1b810ef3a52acbe7fab63",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit-uscold-laredo_2022-09-15/323be8174bcd833a742254ae6f2310a3373007b95de1b810ef3a52acbe7fab63/voxel_safetyvest_vit-uscold-laredo_2022-09-15.tar.gz",
    )

    #Nasha: Bfs/National City Vest Classifier
    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vit-dixieline_2022-09-16",
        sha256 = "2b5f7ad8b295fceb9f968d44317258bd504006d8d5a4e4b0323e4900a2334aa3",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit-dixieline_2022-09-16/2b5f7ad8b295fceb9f968d44317258bd504006d8d5a4e4b0323e4900a2334aa3/voxel_safetyvest_vit-dixieline_2022-09-16.tar.gz",
    )

    # Vivek: Trieagle/Ocala YOLO model 2022/09/19
    voxel_artifact(
        name = "artifacts_2022-09-19-01-24-03-8682-yolo-torchscript",
        sha256 = "1e2bfb5a87a6ffd34d28545d774382762bca7e12e45db93cf55f845e66d6aef5",
        url = "s3://voxel-storage/artifactory/2022-09-19-01-24-03-8682-yolo/1e2bfb5a87a6ffd34d28545d774382762bca7e12e45db93cf55f845e66d6aef5/2022-09-19-01-24-03-8682-yolo.tar.gz",
    )

    # Mamrez: Initial Spill Model trained on real data (laredo/room_f1)
    voxel_artifact(
        name = "artifacts_spill_model_realdata",
        sha256 = "d8c4e3139ebb74894bfba6db98f56d189a9c7b9d9db619de2e5cc15f87ca1df0",
        url = "s3://voxel-storage/artifactory/spill_model_realdata/d8c4e3139ebb74894bfba6db98f56d189a9c7b9d9db619de2e5cc15f87ca1df0/spill_model_realdata.tar.gz",
    )

    # Mamrez: Initial Spill Model trained on Synthetic data V5 (laredo/room_f1)
    voxel_artifact(
        name = "artifacts_spill_model_syntheticdata_v5",
        sha256 = "db9e4c14e60371e6cfd42626515c59db56c7c0310b02adb78c6228e3d7db4ef3",
        url = "s3://voxel-storage/artifactory/spill_model_syntheticdata_v5/db9e4c14e60371e6cfd42626515c59db56c7c0310b02adb78c6228e3d7db4ef3/spill_model_syntheticdata_v5.tar.gz",
    )

    # Vivek/Nasha: Generalized safetyvest model 2022/09/21
    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vit_general_2022-09-21",
        sha256 = "a26cea39212a9bdf1f6cbfa51743721e3e69b75f3fce4dd7237d4665c5e2b4f6",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_general_2022-09-21/a26cea39212a9bdf1f6cbfa51743721e3e69b75f3fce4dd7237d4665c5e2b4f6/voxel_safetyvest_vit_general_2022-09-21.tar.gz",
    )

    # Vivek: Hensley Phoenix and Chandler YOLO model 2022/10/05
    voxel_artifact(
        name = "artifacts_2022-10-04-18-20-08-f6eb-yolo-torchscript",
        sha256 = "f8bb37aa040ca93c7a6586f0d25a9145d45910782448c7b8f064667d41e330d5",
        url = "s3://voxel-storage/artifactory/2022-10-04-18-20-08-f6eb-yolo/f8bb37aa040ca93c7a6586f0d25a9145d45910782448c7b8f064667d41e330d5/2022-10-04-18-20-08-f6eb-yolo.tar.gz",
    )

    # Nasha: Verst/Walton custom safety vest model 2022/10/06
    voxel_artifact(
        name = "artifacts_voxel_safetyvest_vverst-walton_2022-10-06",
        sha256 = "c49d96a7d41220c50ecf68cfea4599fdd44614f4d33e7dedaf9d2d8d619857b2",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vverst-walton_2022-10-06/c49d96a7d41220c50ecf68cfea4599fdd44614f4d33e7dedaf9d2d8d619857b2/voxel_safetyvest_vverst-walton_2022-10-06.tar.gz",
    )

    # Vivek: Feb Distributing YOLO model 2022/10/10
    voxel_artifact(
        name = "artifacts_2022-10-07-20-23-21-2564-yolo-torchscript",
        sha256 = "06385a243ba9fc5136f955841e93fb0e2362ef33df4493d3102a37b177de0b54",
        url = "s3://voxel-storage/artifactory/2022-10-07-20-23-21-2564-yolo/06385a243ba9fc5136f955841e93fb0e2362ef33df4493d3102a37b177de0b54/2022-10-07-20-23-21-2564-yolo.tar.gz",
    )

    # Mamrez: Resnet18 Spill Model trained on Synthetic data V8 (laredo/dock and door)
    voxel_artifact(
        name = "artifacts_resnet18_uscold_laredo_dock",
        sha256 = "c3b9e60aea5a2940d4104e700b99c2397e7457c6af7775b3347de61fe866c43d",
        url = "s3://voxel-storage/artifactory/resnet18_uscold_laredo_dock/c3b9e60aea5a2940d4104e700b99c2397e7457c6af7775b3347de61fe866c43d/resnet18_uscold_laredo_dock.tar.gz",
    )

    # Mamrez: Resnet18 Spill Model trained on Synthetic data V8 (laredo/room)
    voxel_artifact(
        name = "artifacts_resnet18_uscold_laredo_room",
        sha256 = "fd90ad8bb344a662dab61bce688c3b77b775516a9e4a06c44a9c8cdfac2cdaae",
        url = "s3://voxel-storage/artifactory/resnet18_uscold_laredo_room/fd90ad8bb344a662dab61bce688c3b77b775516a9e4a06c44a9c8cdfac2cdaae/resnet18_uscold_laredo_room.tar.gz",
    )

    # Vivek: Americold/Taunton Door Model 0003 2022-10-21
    voxel_artifact(
        name = "artifacts_2022-10-21-T16-34americold_taunton_0003_cha",
        sha256 = "587edadd96e2465e06258e421e977dbb35a91a1bdd9a44e8e8cae8dc9a534818",
        url = "s3://voxel-storage/artifactory/2022-10-21-T16-34americold_taunton_0003_cha/587edadd96e2465e06258e421e977dbb35a91a1bdd9a44e8e8cae8dc9a534818/2022-10-21-T16-34americold_taunton_0003_cha.tar.gz",
    )

    # Vivek: Americold/Tacoma Door Model 0004 2022-10-23
    voxel_artifact(
        name = "artifacts_2022-10-25-T19-22americold_tacoma_0004_cha",
        sha256 = "32cbb8a745f8d37d9268431e853ae400ea3d438a5a9281aa9e1d09edd67a15a3",
        url = "s3://voxel-storage/artifactory/2022-10-25-T19-22americold_tacoma_0004_cha/32cbb8a745f8d37d9268431e853ae400ea3d438a5a9281aa9e1d09edd67a15a3/2022-10-25-T19-22americold_tacoma_0004_cha.tar.gz",
    )

    # Vivek: Americold/Tacoma Door Model 0005 2023-02-15 (After DoorDataflywheel)
    voxel_artifact(
        name = "artifacts_02_15_2023_americold_tacoma_0005_cha",
        sha256 = "9a4ef91e5914b4bde91970f1a9d2cad4f12d8334c998561ec4ac92d07629ebb8",
        url = "s3://voxel-storage/artifactory/8fcfdb4e-f2aa-476d-b2e5-f7347755d1f2/9a4ef91e5914b4bde91970f1a9d2cad4f12d8334c998561ec4ac92d07629ebb8/8fcfdb4e-f2aa-476d-b2e5-f7347755d1f2.tar.gz",
    )

    # Vivek: Americold/Ontario Door Model 0003 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_ontario_0003_cha",
        sha256 = "2f6fa31060c703694b9144ec1b63df36d2907df5bbfd03cffb5c032fb6a370e8",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_ontario_0003_cha/2f6fa31060c703694b9144ec1b63df36d2907df5bbfd03cffb5c032fb6a370e8/2022-10-26_americold_ontario_0003_cha.tar.gz",
    )

    # Vivek: Americold/Ontario Door Model 0009 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_ontario_0009_cha",
        sha256 = "1f4802e66e9528c669c33ed97e2498b763a138a8ffadcd6302e29592f7f437c7",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_ontario_0009_cha/1f4802e66e9528c669c33ed97e2498b763a138a8ffadcd6302e29592f7f437c7/2022-10-26_americold_ontario_0009_cha.tar.gz",
    )

    # Vivek: Americold/Ontario Door Model 0011 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_ontario_0011_cha",
        sha256 = "46a4f27f62ca6eb76f5b4c947130f7b420dc3049486d942273d597ac19662710",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_ontario_0011_cha/46a4f27f62ca6eb76f5b4c947130f7b420dc3049486d942273d597ac19662710/2022-10-26_americold_ontario_0011_cha.tar.gz",
    )

    # Vivek: Americold/Ontario Door Model 0012 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_ontario_0012_cha",
        sha256 = "aa24db93161f37326e01f77b593d831c99cdcd9186ea6d6fc1c63c574f96c9b2",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_ontario_0012_cha/aa24db93161f37326e01f77b593d831c99cdcd9186ea6d6fc1c63c574f96c9b2/2022-10-26_americold_ontario_0012_cha.tar.gz",
    )

    # Vivek: Americold/Modesto Door Model 0001 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_modesto_0001_cha",
        sha256 = "4bea97bb362b26126371d12f9360b9cfd565124d8e9bb5dfc4906138223f059e",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_modesto_0001_cha/4bea97bb362b26126371d12f9360b9cfd565124d8e9bb5dfc4906138223f059e/2022-10-26_americold_modesto_0001_cha.tar.gz",
    )

    # Vivek: Americold/Modesto Door Model 0002 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_modesto_0002_cha",
        sha256 = "83f6d04ccc9cb11c3d8f9831b23dfdc8d8bd5dafef4c0e49be5ff7a1abf0658c",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_modesto_0002_cha/83f6d04ccc9cb11c3d8f9831b23dfdc8d8bd5dafef4c0e49be5ff7a1abf0658c/2022-10-26_americold_modesto_0002_cha.tar.gz",
    )

    # Vivek: Americold/Modesto Door Model 0003 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_modesto_0003_cha",
        sha256 = "8211d8929a0d9edb6c9c91b40f2781804e97f5806d48d103e3651e7fd991a7b5",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_modesto_0003_cha/8211d8929a0d9edb6c9c91b40f2781804e97f5806d48d103e3651e7fd991a7b5/2022-10-26_americold_modesto_0003_cha.tar.gz",
    )

    # Vivek: Americold/Modesto Door Model 0005 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_modesto_0005_cha",
        sha256 = "a2655c502b1b4db27899622f7e79c8fee834b346bc808d006dba625af0c36a4e",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_modesto_0005_cha/a2655c502b1b4db27899622f7e79c8fee834b346bc808d006dba625af0c36a4e/2022-10-26_americold_modesto_0005_cha.tar.gz",
    )

    # Vivek: Americold/Modesto Door Model 0009 2022-10-26 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-26_americold_modesto_0009_cha",
        sha256 = "3e6eda95691ad9ecd72c51c064c798f592f39113789ebdf11975effd8dc10ab4",
        url = "s3://voxel-storage/artifactory/2022-10-26_americold_modesto_0009_cha/3e6eda95691ad9ecd72c51c064c798f592f39113789ebdf11975effd8dc10ab4/2022-10-26_americold_modesto_0009_cha.tar.gz",
    )

    # Vivek: Americold/Sanford Door Model 0001 2022-10-28 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-28_americold_sanford_0001_cha",
        sha256 = "02ec597a4f7f69669c8869a5c5dc054597f6e1c187d384aab6800f9dfda0c29b",
        url = "s3://voxel-storage/artifactory/2022-10-28_americold_sanford_0001_cha/02ec597a4f7f69669c8869a5c5dc054597f6e1c187d384aab6800f9dfda0c29b/2022-10-28_americold_sanford_0001_cha.tar.gz",
    )

    # Vivek: Americold/Sanford Door Model 0005 2022-10-28 - Removed previous model
    voxel_artifact(
        name = "artifacts_2022-10-28_americold_sanford_0005_cha",
        sha256 = "53433ecc53631952d291f50e0fc7fffe8fbfbaf8b1168a973187f79732a0b171",
        url = "s3://voxel-storage/artifactory/2022-10-28_americold_sanford_0005_cha/53433ecc53631952d291f50e0fc7fffe8fbfbaf8b1168a973187f79732a0b171/2022-10-28_americold_sanford_0005_cha.tar.gz",
    )

    # Vivek: Americold/Savannah_Pooler Door Model 0006 2022-10-28 - Only this model exists
    voxel_artifact(
        name = "artifacts_2022-10-28_americold_savannah_pooler_0006_cha",
        sha256 = "373918bfaf1befd558135ce54839843049ebffe8a635ce750d88c463de5857c3",
        url = "s3://voxel-storage/artifactory/2022-10-28_americold_savannah_pooler_0006_cha/373918bfaf1befd558135ce54839843049ebffe8a635ce750d88c463de5857c3/2022-10-28_americold_savannah_pooler_0006_cha.tar.gz",
    )

    # Vivek: Americold/Taunton Door Model 0006 2022-10-28 - Only this model exists for taunton/0006
    voxel_artifact(
        name = "artifacts_2022-10-28_americold_taunton_0006_cha",
        sha256 = "edbf0df85e1a2932dc19314be0fd15462102c0e1c962d3a3a45246a2f560b2b5",
        url = "s3://voxel-storage/artifactory/2022-10-28_americold_taunton_0006_cha/edbf0df85e1a2932dc19314be0fd15462102c0e1c962d3a3a45246a2f560b2b5/2022-10-28_americold_taunton_0006_cha.tar.gz",
    )

    # Vivek: Americold/Taunton Door Model 0007 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_americold_taunton_0007_cha",
        sha256 = "e51425ce63b1ca69ff54f4325abd56d8da7b05abcd7f563506e50ac52e8e0b24",
        url = "s3://voxel-storage/artifactory/2022-10-28_americold_taunton_0007_cha/e51425ce63b1ca69ff54f4325abd56d8da7b05abcd7f563506e50ac52e8e0b24/2022-10-28_americold_taunton_0007_cha.tar.gz",
    )

    # Vivek: Americold/Taunton Door Model 0010 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_americold_taunton_0010_cha",
        sha256 = "bb2e177f951eea6e78d4b8f75696037632aed97a7069d532cad79d20478b11aa",
        url = "s3://voxel-storage/artifactory/2022-10-28_americold_taunton_0010_cha/bb2e177f951eea6e78d4b8f75696037632aed97a7069d532cad79d20478b11aa/2022-10-28_americold_taunton_0010_cha.tar.gz",
    )

    # Vivek: BFS/Dixieline Door Model 0001 2023-02-15 - (After Doordataflywheel for night corrupted images)
    voxel_artifact(
        name = "artifacts_02_15_2023_buildersfirstsource_dixieline_0001_cha",
        sha256 = "e814dd831642a26a281f77f499ab7676c4f16a04dc4751a215adf9a76c56a84f",
        url = "s3://voxel-storage/artifactory/855251d5-940a-4977-8ae2-28ff961239cb/e814dd831642a26a281f77f499ab7676c4f16a04dc4751a215adf9a76c56a84f/855251d5-940a-4977-8ae2-28ff961239cb.tar.gz",
    )

    # Vivek: FEB_distributing/Biloxi Door Model 0005 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_feb_distributing_biloxi_0005_cha",
        sha256 = "25b7ea7eb87677d91395e9290dcb3e9baf855baa7fbad99d5200db5969e4737b",
        url = "s3://voxel-storage/artifactory/2022-10-28_feb_distributing_biloxi_0005_cha/25b7ea7eb87677d91395e9290dcb3e9baf855baa7fbad99d5200db5969e4737b/2022-10-28_feb_distributing_biloxi_0005_cha.tar.gz",
    )

    # Vivek: Hensley/Chandler Door Model 0005 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_hensley_chandler_0005_cha",
        sha256 = "d6faa82fd399e62105a43bca85ffb1235d5236e46497dc5fc93986cc93ae4b39",
        url = "s3://voxel-storage/artifactory/2022-10-28_hensley_chandler_0005_cha/d6faa82fd399e62105a43bca85ffb1235d5236e46497dc5fc93986cc93ae4b39/2022-10-28_hensley_chandler_0005_cha.tar.gz",
    )

    # Vivek: Hensley/Phoenix Door Model 0003 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_hensley_phoenix_0003_cha",
        sha256 = "b94457925b74b889af431256ce7f14b7a791799f206f4eeb2e7096e36315fa9a",
        url = "s3://voxel-storage/artifactory/2022-10-28_hensley_phoenix_0003_cha/b94457925b74b889af431256ce7f14b7a791799f206f4eeb2e7096e36315fa9a/2022-10-28_hensley_phoenix_0003_cha.tar.gz",
    )

    # Vivek: Trieagle/Ocala Door Model 0001 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_trieagle_ocala_0001_cha",
        sha256 = "c50e1ee073363607e7d0f7f55a0cb059b7bf9ef983cc3c09a945760d903b670b",
        url = "s3://voxel-storage/artifactory/2022-10-28_trieagle_ocala_0001_cha/c50e1ee073363607e7d0f7f55a0cb059b7bf9ef983cc3c09a945760d903b670b/2022-10-28_trieagle_ocala_0001_cha.tar.gz",
    )

    # Vivek: Trieagle/Ocala Door Model 0005 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_trieagle_ocala_0005_cha",
        sha256 = "957f909299cdf92689e989b1fce7a59831ba83b1efe65b817b0739904476d8bb",
        url = "s3://voxel-storage/artifactory/2022-10-28_trieagle_ocala_0005_cha/957f909299cdf92689e989b1fce7a59831ba83b1efe65b817b0739904476d8bb/2022-10-28_trieagle_ocala_0005_cha.tar.gz",
    )

    # Vivek: Uscold/Quakertown Door Model 0002 2022-10-28 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-28_uscold_quakertown_0002_cha",
        sha256 = "ad83137a92646db5ca59cafde7da484b95926d59392291aa265c33495439d9ec",
        url = "s3://voxel-storage/artifactory/2022-10-28_uscold_quakertown_0002_cha/ad83137a92646db5ca59cafde7da484b95926d59392291aa265c33495439d9ec/2022-10-28_uscold_quakertown_0002_cha.tar.gz",
    )

    # Vivek: Americold/Ontario Door Model 0001 2022-10-31 - Removed older model
    voxel_artifact(
        name = "artifacts_2022-10-31_americold_ontario_0001_cha",
        sha256 = "545509b7df2e45737d023ecd3e921f633b067dca76e9ab85782bcf515dfa5b85",
        url = "s3://voxel-storage/artifactory/2022-10-31_americold_ontario_0001_cha/545509b7df2e45737d023ecd3e921f633b067dca76e9ab85782bcf515dfa5b85/2022-10-31_americold_ontario_0001_cha.tar.gz",
    )

    # Gabriel: TRT Version 8.4.1.5 YOLO models exported
    voxel_artifact(
        name = "artifacts_2021-12-06-00-00-00-0000-yolo",
        sha256 = "740b8f882d9a83603f2991b17ce010178c4c859a897f4321923d8a26aef42b6d",
        url = "s3://voxel-storage/artifactory/2021-12-06-00-00-00-0000-yolo/740b8f882d9a83603f2991b17ce010178c4c859a897f4321923d8a26aef42b6d/2021-12-06-00-00-00-0000-yolo.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_2022-08-31-01-26-05-aa61-yolo",
        sha256 = "3b2f935573cf8873f607ecb92d693b90c28893af3b535043b5de7f506e9ca99e",
        url = "s3://voxel-storage/artifactory/2022-08-31-01-26-05-aa61-yolo/3b2f935573cf8873f607ecb92d693b90c28893af3b535043b5de7f506e9ca99e/2022-08-31-01-26-05-aa61-yolo.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_2022-09-19-01-24-03-8682-yolo",
        sha256 = "67161c4998443c5336c3698c530667c22323c8b9693865e11f00ffd59752ac8d",
        url = "s3://voxel-storage/artifactory/2022-09-19-01-24-03-8682-yolo/67161c4998443c5336c3698c530667c22323c8b9693865e11f00ffd59752ac8d/2022-09-19-01-24-03-8682-yolo.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_2022-10-04-18-20-08-f6eb-yolo",
        sha256 = "700668bcff6e51d7e96ef4618df8b044877a671c64f58027bf6fff3f2c833459",
        url = "s3://voxel-storage/artifactory/2022-10-04-18-20-08-f6eb-yolo/700668bcff6e51d7e96ef4618df8b044877a671c64f58027bf6fff3f2c833459/2022-10-04-18-20-08-f6eb-yolo.tar.gz",
    )

    voxel_artifact(
        name = "artifacts_2022-10-07-20-23-21-2564-yolo",
        sha256 = "4f291b25816c8b0ac94d2823af06de3ad56b7740e0fa2be623de5c09f8adf8fd",
        url = "s3://voxel-storage/artifactory/2022-10-07-20-23-21-2564-yolo/4f291b25816c8b0ac94d2823af06de3ad56b7740e0fa2be623de5c09f8adf8fd/2022-10-07-20-23-21-2564-yolo.tar.gz",
    )

    # Mamrez: Resnet18 Spill Model trained on Synthetic data V4-V8 (laredo/room) binary smp#
    voxel_artifact(
        name = "artifacts_resnet18_uscold_laredo_room_smp_binary",
        sha256 = "f1c9188c226de84a53f1d22d1917fdabb49b3bedcd00b8e8299d9f78c7dd8f59",
        url = "s3://voxel-storage/artifactory/resnet18_uscold_laredo_room_smp_binary/f1c9188c226de84a53f1d22d1917fdabb49b3bedcd00b8e8299d9f78c7dd8f59/resnet18_uscold_laredo_room_smp_binary.tar.gz",
    )

    # Mamrez: Resnet18 Spill Model trained on Synthetic data 8 (laredo/dock door) binary smp#
    voxel_artifact(
        name = "artifacts_resnet18_uscold_laredo_dock_smp_binary",
        sha256 = "6fd4c3cd1689aec0629b20a8dd64056076a396e63593cb2fc9705f8fc7157645",
        url = "s3://voxel-storage/artifactory/resnet18_uscold_laredo_dock_smp_binary/6fd4c3cd1689aec0629b20a8dd64056076a396e63593cb2fc9705f8fc7157645/resnet18_uscold_laredo_dock_smp_binary.tar.gz",
    )

    # Mamrez: Resnet18 Spill Model trained on Synthetic data 8 (laredo/dock door) binary smp#
    voxel_artifact(
        name = "artifacts_resnet18_uscold_laredo_alldock_smp_binary",
        sha256 = "cf5b1bc72953d38a030b90571d5e61525353be0d6378c6c2a62b1a302c02cc8b",
        url = "s3://voxel-storage/artifactory/resnet18_uscold_laredo_alldock_smp_binary/cf5b1bc72953d38a030b90571d5e61525353be0d6378c6c2a62b1a302c02cc8b/resnet18_uscold_laredo_alldock_smp_binary.tar.gz",
    )

    # Vivek: americold_savannah_bloomingdale_0006 Door dataflywheel trained 2022-11-08
    voxel_artifact(
        name = "artifacts_2022-11-08_americold_savannah_bloomingdale_0006_cha",
        sha256 = "20e04c5c79cb9903e22a4dc6b14913459949edf68f192ef61b99c7c9d4eacdf0",
        url = "s3://voxel-storage/artifactory/2022-11-08_americold_savannah_bloomingdale_0006_cha/20e04c5c79cb9903e22a4dc6b14913459949edf68f192ef61b99c7c9d4eacdf0/2022-11-08_americold_savannah_bloomingdale_0006_cha.tar.gz",
    )

    # Vivek: builderfirstsource_solanabeach_0003_cha Door dataflywheel trained 2022-11-08
    voxel_artifact(
        name = "artifacts_2022-11-08_builderfirstsource_solanabeach_0003_cha",
        sha256 = "2257b1829f0363fff742324fab48d4646a9663eae2df2017bd7c434ff8648118",
        url = "s3://voxel-storage/artifactory/2022-11-08_builderfirstsource_solanabeach_0003_cha/2257b1829f0363fff742324fab48d4646a9663eae2df2017bd7c434ff8648118/2022-11-08_builderfirstsource_solanabeach_0003_cha.tar.gz",
    )

    # Vivek: hensley_phoenix_0002_cha Door dataflywheel trained 2022-11-08
    voxel_artifact(
        name = "artifacts_2022-11-08_hensley_phoenix_0002_cha",
        sha256 = "7c7ad3bb6b0eb536ac62f146e01b8aded028e6a0eb939016091e6f52758db5d1",
        url = "s3://voxel-storage/artifactory/2022-11-08_hensley_phoenix_0002_cha/7c7ad3bb6b0eb536ac62f146e01b8aded028e6a0eb939016091e6f52758db5d1/2022-11-08_hensley_phoenix_0002_cha.tar.gz",
    )

    # Vivek: Yolov5 TRT 2022-11-01 for lakeshore_beverage/arlington_heights & feb_distributing/biloxi
    voxel_artifact(
        name = "artifacts_2022-11-01-06-31-22-ef98-yolo",
        sha256 = "ddf2a0c42846fe751f81179ac544e1deda82bce45389310fb3b35fbe230cec27",
        url = "s3://voxel-storage/artifactory/2022-11-01-06-31-22-ef98-yolo/ddf2a0c42846fe751f81179ac544e1deda82bce45389310fb3b35fbe230cec27/2022-11-01-06-31-22-ef98-yolo.tar.gz",
    )

    # Vivek: Yolov5 TRT for americold/tacoma
    voxel_artifact(
        name = "artifacts_2022-11-09-22-58-28-2d35-yolo",
        sha256 = "9a2e655e7a0eaf8302b77f8176a3e75e2b87314fcf69797aacac6f725ee88d05",
        url = "s3://voxel-storage/artifactory/2022-11-09-22-58-28-2d35-yolo/9a2e655e7a0eaf8302b77f8176a3e75e2b87314fcf69797aacac6f725ee88d05/2022-11-09-22-58-28-2d35-yolo.tar.gz",
    )

    # Vivek: Yolov5 TRT for wn_foods/hayward 2022-11-15
    voxel_artifact(
        name = "artifacts_2022-11-15-03-38-16-91db-yolo",
        sha256 = "6f984c2f403ba1f589f9197ecc86bd53a023c30eafe4a7d40c2cd3d870b70ee4",
        url = "s3://voxel-storage/artifactory/2022-11-15-03-38-16-91db-yolo/6f984c2f403ba1f589f9197ecc86bd53a023c30eafe4a7d40c2cd3d870b70ee4/2022-11-15-03-38-16-91db-yolo.tar.gz",
    )

    # Vivek Hard_hat ViTv4 after training on 40k images (with Occlusion)
    voxel_artifact(
        name = "artifacts_hard_hat_occlusion_vit_v4_2022-11-18",
        sha256 = "47d30cd2bb3872ee06e47d65ed24a4c91b69f215c18c4aaa19f02fc0ff386134",
        url = "s3://voxel-storage/artifactory/hard_hat_occlusion_vit_v4_2022-11-18/47d30cd2bb3872ee06e47d65ed24a4c91b69f215c18c4aaa19f02fc0ff386134/hard_hat_occlusion_vit_v4_2022-11-18.tar.gz",
    )

    # Mamrez: Spill Model gen_dock_door_v1
    voxel_artifact(
        name = "artifacts_smp-spills-smp-all-dock-845d3bb3-1296-49",
        sha256 = "91a3e8bd8c1cd3ca65b1e28ac0c5380f287c5429572595e0d1b455d428c1c0f3",
        url = "s3://voxel-storage/artifactory/smp-spills-smp-all-dock-845d3bb3-1296-49/91a3e8bd8c1cd3ca65b1e28ac0c5380f287c5429572595e0d1b455d428c1c0f3/smp-spills-smp-all-dock-845d3bb3-1296-49.tar.gz",
    )

    # Mamrez: Spill Model gen_room_v1
    voxel_artifact(
        name = "artifacts_smp-spills-smp-b908a959-1880-45",
        sha256 = "0aa69e8a6c20925b5ed7c8131ee8e6662fb008d11b87a824b486addfabe5e9c7",
        url = "s3://voxel-storage/artifactory/smp-spills-smp-b908a959-1880-45/0aa69e8a6c20925b5ed7c8131ee8e6662fb008d11b87a824b486addfabe5e9c7/smp-spills-smp-b908a959-1880-45.tar.gz",
    )

    # Vivek: bumpcapv1 - 14k barehead vs 20k non-barehead
    voxel_artifact(
        name = "artifacts_bumpcap_vit_99_46_34k_2022-12-20",
        sha256 = "5bd13f824e0be6300871d0d284ab885349f50975ec8a7fd21fcfcc7fac0a1b51",
        url = "s3://voxel-storage/artifactory/bumpcap_vit_99_46_34k_2022-12-20/5bd13f824e0be6300871d0d284ab885349f50975ec8a7fd21fcfcc7fac0a1b51/bumpcap_vit_99_46_34k_2022-12-20.tar.gz",
    )

    # Vivek: Adding door model for wn_foods/hayward/0012/cha - 2023/01/09
    voxel_artifact(
        name = "artifacts_01_09_2023_wn_foods_hayward_0012_cha",
        sha256 = "9eca18e953168ea0bb16d1cbf604d473e54e1876777b3d90be86d7786fad4304",
        url = "s3://voxel-storage/artifactory/860ebac6-dc32-43ed-91dd-a6b61bc04dce/9eca18e953168ea0bb16d1cbf604d473e54e1876777b3d90be86d7786fad4304/860ebac6-dc32-43ed-91dd-a6b61bc04dce.tar.gz",
    )

    # Vivek: Adding door model for wn_foods/hayward/0009/cha - 2023/01/09
    voxel_artifact(
        name = "artifacts_01_09_2023_wn_foods_hayward_0009_cha",
        sha256 = "2b206bba0b622e05394ddabaec790e2ee2014c2f7f23d03bf5a8b8f8fcf9f8a6",
        url = "s3://voxel-storage/artifactory/8380802c-969f-44a0-b909-72c83fa2bb66/2b206bba0b622e05394ddabaec790e2ee2014c2f7f23d03bf5a8b8f8fcf9f8a6/8380802c-969f-44a0-b909-72c83fa2bb66.tar.gz",
    )

    # Vivek: Adding door model for wn_foods/hayward/0011/cha - 2023/02/15 (After Door Dataflywheel)
    voxel_artifact(
        name = "artifacts_02_15_2023_wn_foods_hayward_0011_cha",
        sha256 = "71d23b2b5ca96a8e0a5d5d7c05aad2a1f9c2f3a63c23df9ac34b3f55e58ef256",
        url = "s3://voxel-storage/artifactory/9b4a0f9b-5c33-4af2-b6d1-238ee9b30232/71d23b2b5ca96a8e0a5d5d7c05aad2a1f9c2f3a63c23df9ac34b3f55e58ef256/9b4a0f9b-5c33-4af2-b6d1-238ee9b30232.tar.gz",
    )

    # Vivek: Adding Yolov5 TRT model for ppg/cedar_falls, western_carriers/north_bergen, piston_automotive/marion, and innovate_manufacturing/knoxville
    voxel_artifact(
        name = "artifacts_01_18_2023_piston-automotive-yolo",
        sha256 = "51819598f8dab67b31e7a3e584fb22180b582c600541b56af593e48e700bc732",
        url = "s3://voxel-storage/artifactory/9623a6bd5d4e42df8b73208869a1072b-yolo/51819598f8dab67b31e7a3e584fb22180b582c600541b56af593e48e700bc732/9623a6bd5d4e42df8b73208869a1072b-yolo.tar.gz",
    )

    # Vivek: Adding Generalized EXIT Door (Used for Wesco/reno currently) - 96% Recall
    voxel_artifact(
        name = "artifacts_02_06_2023_generalized_EXIT_Door_new",
        sha256 = "b819aa4a6af23b161203a5146c4b8153c40f58c0508a763aa52ea9632585eb25",
        url = "s3://voxel-storage/artifactory/7547a635-1ae9-4b9a-b44a-4e8991269ead/b819aa4a6af23b161203a5146c4b8153c40f58c0508a763aa52ea9632585eb25/7547a635-1ae9-4b9a-b44a-4e8991269ead.tar.gz",
    )

    # Vivek: Adding Generalized FRONT Door (Used for innovate_manufacturing/knoxville/0005/cha currently)
    voxel_artifact(
        name = "artifacts_01_18_2023_generalized-front-door",
        sha256 = "81d808a23bb58de0fb52b5d2e9520ce5508d3566361af20c66fa25cad363672f",
        url = "s3://voxel-storage/artifactory/3f11d513-b9bd-471e-862e-4d2474fc443c/81d808a23bb58de0fb52b5d2e9520ce5508d3566361af20c66fa25cad363672f/3f11d513-b9bd-471e-862e-4d2474fc443c.tar.gz",
    )

    # Vivek: Adding Old Generalized SIDE Door (Used for innovate_manufacturing/knoxville/0004/cha currently)
    voxel_artifact(
        name = "artifacts_02_09_2023_old_generalized_side_door",
        url = "s3://voxel-storage/artifactory/91f587b1-a464-4bb2-9124-4634a12ee929/8aefdbdd3d8344f06f6370ef1bfb1b910455f3963265a71da758d42529860534/91f587b1-a464-4bb2-9124-4634a12ee929.tar.gz",
        sha256 = "8aefdbdd3d8344f06f6370ef1bfb1b910455f3963265a71da758d42529860534",
    )

    # Vivek: Adding Innovate_manufacturing/knxoville specialized model
    voxel_artifact(
        name = "artifacts_02_15_2023_innovate_manufacturing_knoxville_0004_cha",
        sha256 = "e0753729159dd62c3654adfdeeffb3f56e8c1546712b6094cb5f8daf16f2eb59",
        url = "s3://voxel-storage/artifactory/6ec9304f-e1be-4731-b687-82c318bfbf6d/e0753729159dd62c3654adfdeeffb3f56e8c1546712b6094cb5f8daf16f2eb59/6ec9304f-e1be-4731-b687-82c318bfbf6d.tar.gz",
    )

    # Mamrez: spill generalized model
    voxel_artifact(
        name = "artifacts_02_05_2023_spill_generalized",
        sha256 = "02da2ac21a8d3276eb6897ee38176498630731f976da0d7502515baab0027221",
        url = "s3://voxel-storage/artifactory/smp-bd6b62dfd5b64dcfb4970102e2c9b2aa/02da2ac21a8d3276eb6897ee38176498630731f976da0d7502515baab0027221/smp-bd6b62dfd5b64dcfb4970102e2c9b2aa.tar.gz",
    )

    # Vivek: Adding americold/ontario/0004 After dataflywheel and new data ingestion
    voxel_artifact(
        name = "artifacts_02_24_2023_americold_ontario_0004_cha",
        sha256 = "ec45cf2d59aed51435d59045951f4ebc49493bffeee2846ef9f822d32505804f",
        url = "s3://voxel-storage/artifactory/1ce53a31-91d4-4d1c-b24a-87d30e88d689/ec45cf2d59aed51435d59045951f4ebc49493bffeee2846ef9f822d32505804f/1ce53a31-91d4-4d1c-b24a-87d30e88d689.tar.gz",
    )

    # Vivek: Adding americold/ontario/0005 before dataflywheel
    voxel_artifact(
        name = "artifacts_02_15_2023_americold_ontario_0005_cha",
        sha256 = "e75ffb4aa98c2b33b04d8931aa4862eff44df1809dae546f2a3f0cc4dc678587",
        url = "s3://voxel-storage/artifactory/54dedcff-fcc9-4521-927b-9c9f3114b320/e75ffb4aa98c2b33b04d8931aa4862eff44df1809dae546f2a3f0cc4dc678587/54dedcff-fcc9-4521-927b-9c9f3114b320.tar.gz",
    )

    # Vivek: Adding americold/ontario/0006 before dataflywheel
    voxel_artifact(
        name = "artifacts_02_15_2023_americold_ontario_0006_cha",
        sha256 = "49e3367d197a11803d600468c9f1418dba5decce4bd6273a01147b96fa5c1296",
        url = "s3://voxel-storage/artifactory/b623a0be-0dbf-40c9-ab03-e3c6f1324d21/49e3367d197a11803d600468c9f1418dba5decce4bd6273a01147b96fa5c1296/b623a0be-0dbf-40c9-ab03-e3c6f1324d21.tar.gz",
    )

    # Vivek: Adding new yolo model trained on wesco/reno, office_depot, and michaels/tracy
    voxel_artifact(
        name = "artifacts_02_27_2023_michaels_wesco_office_yolo",
        sha256 = "37a89385b644e890dd756ac2fa032ff535009b152932257e1107beaab762baf1",
        url = "s3://voxel-storage/artifactory/best_736_1280/37a89385b644e890dd756ac2fa032ff535009b152932257e1107beaab762baf1/best_736_1280.tar.gz",
    )

    # Vivek: Adding generalized freezer doors used for americold/ontario_bldg_2 cameras 3, 4, & 5
    voxel_artifact(
        name = "artifacts_03_04_2023_generalized_freezer_door",
        url = "s3://voxel-storage/artifactory/c89809bd-ddad-451b-8e5e-c1e7d3434b41/a95ac344959e396dfe100749d4f9fa8c1f1f924cab015e1786579c5bc96f91b4/c89809bd-ddad-451b-8e5e-c1e7d3434b41.tar.gz",
        sha256 = "a95ac344959e396dfe100749d4f9fa8c1f1f924cab015e1786579c5bc96f91b4",
    )

    # Vivek: Adding new modesto 11 door trained after dataflywheel
    voxel_artifact(
        name = "artifacts_03_05_2023_americold_modesto_0011_cha",
        url = "s3://voxel-storage/artifactory/256037fe-697a-4484-ab48-8c81b6ea18e1/93ef0ec78300f77e54bc01907dafa1389a6ce2e64d8652884798740be2e19298/256037fe-697a-4484-ab48-8c81b6ea18e1.tar.gz",
        sha256 = "93ef0ec78300f77e54bc01907dafa1389a6ce2e64d8652884798740be2e19298",
    )

    # Vivek: Adding new fine-tuned generalized freezer model model
    voxel_artifact(
        name = "artifacts_03_13_2023_americold_ontario_0104_fine_tuned_freezer_generalized_model",
        url = "s3://voxel-storage/artifactory/7df6fe0c-5e28-43f0-b375-6b231975cfff/a60bd885389beaf238f20acdc2f6a410511b9facecc9a1f024d30a7cab0f8026/7df6fe0c-5e28-43f0-b375-6b231975cfff.tar.gz",
        sha256 = "a60bd885389beaf238f20acdc2f6a410511b9facecc9a1f024d30a7cab0f8026",
    )

    # Vai: Add carry object classifier
    voxel_artifact(
        name = "artifacts_03_24_2023_carry_classifier",
        url = "s3://voxel-storage/artifactory/best_lift_DSv4_RN34/ee35cf90c48a9d38b192b21aeb437e01ae6fc821339906b3f7048b4fafc1acae/best_lift_DSv4_RN34.tar.gz",
        sha256 = "ee35cf90c48a9d38b192b21aeb437e01ae6fc821339906b3f7048b4fafc1acae",
    )
    voxel_artifact(
        name = "artifacts_03_23_2023_overreaching_model_jit",
        url = "s3://voxel-storage/artifactory/voxel_ergo_ml_overreaching_jit/39bc1298403551a6303607f480188b970610480bbd2c22b5c3b3b0455782f0e1/voxel_ergo_ml_overreaching_jit.tar.gz",
        sha256 = "39bc1298403551a6303607f480188b970610480bbd2c22b5c3b3b0455782f0e1",
    )

    voxel_artifact(
        name = "artifacts_02_05_2023_spill_generalized_jit",
        url = "s3://voxel-storage/artifactory/02_05_2023_spill_generalized_jit/42277a61fd909328882541fbdf1846dce7b80b8b2dfab8fdaae46ae475baae41/02_05_2023_spill_generalized_jit.tar.gz",
        sha256 = "42277a61fd909328882541fbdf1846dce7b80b8b2dfab8fdaae46ae475baae41",
    )

    voxel_artifact(
        name = "artifacts_03_21_2023_pose_0630_jit_update",
        url = "s3://voxel-storage/artifactory/pose_0630_jit/0d93afdc0d77eadccaf81f96a3c71b73154be08bd1a7eacc805e200f868830d6/pose_0630_jit.tar.gz",
        sha256 = "0d93afdc0d77eadccaf81f96a3c71b73154be08bd1a7eacc805e200f868830d6",
    )

    # Vivek: Adding new ulta/dallas/0003/cha EXIT door model (trained locally)
    voxel_artifact(
        name = "artifacts_04_03_2023_ulta_dallas_0003_cha_local",
        url = "s3://voxel-storage/artifactory/40f4adfb-733e-4575-8fec-46973c2c34fd/2966422cc37d197bbaf813a49085291c549dd779e95237e5cb24eb75d3ef86ba/40f4adfb-733e-4575-8fec-46973c2c34fd.tar.gz",
        sha256 = "2966422cc37d197bbaf813a49085291c549dd779e95237e5cb24eb75d3ef86ba",
    )

    voxel_artifact(
        name = "artifacts_test_image_inference",
        url = "s3://voxel-storage/artifactory/test_image/95f208b8d3c4ab6c7eec58fa7ba95930a99c17f61e5a08eea77bf623fa260ce8/test_image.tar.gz",
        sha256 = "95f208b8d3c4ab6c7eec58fa7ba95930a99c17f61e5a08eea77bf623fa260ce8",
    )

    # Vivek: Adding new jfe_shoji_burlington_0011_cha DOCK door model
    voxel_artifact(
        name = "artifacts_04_05_2023_jfe_shoji_burlington_0011_cha",
        url = "s3://voxel-storage/artifactory/80fe5e4f-b980-4b09-a037-b80b95c07ef9/8d8af76f9e0a0d9cf363085b2ebbb9fdc8ba0b96ac4fdd20718184e44ee366c9/80fe5e4f-b980-4b09-a037-b80b95c07ef9.tar.gz",
        sha256 = "8d8af76f9e0a0d9cf363085b2ebbb9fdc8ba0b96ac4fdd20718184e44ee366c9",
    )

    voxel_artifact(
        name = "artifacts_yolo_v5_post_processing_04_11_2023",
        url = "s3://voxel-storage/artifactory/yolo_postprocessing_apr11/d3bbc3c0d669c35a72ec9d5041e137f0895159be39fe5308d04a98bcb4bb0962/yolo_postprocessing_apr11.tar.gz",
        sha256 = "d3bbc3c0d669c35a72ec9d5041e137f0895159be39fe5308d04a98bcb4bb0962",
    )

    # Gabriel: Preprocessing takes in NCHW input tensor
    voxel_artifact(
        name = "artifacts_yolo_v5_pre_processing_04_11_2023",
        url = "s3://voxel-storage/artifactory/yolo_preprocessing_apr11/42301d22652b6c32f61432ef05c8b33e9acc3ce7a515fd23602316147901e39b/yolo_preprocessing_apr11.tar.gz",
        sha256 = "42301d22652b6c32f61432ef05c8b33e9acc3ce7a515fd23602316147901e39b",
    )

    # Gabriel: Preprocessing takes in NHWC input tensor
    voxel_artifact(
        name = "artifacts_yolo_v5_pre_processing_04_13_2023",
        url = "s3://voxel-storage/artifactory/yolo_preprocessing_apr13/8438f66f3cf810664190d83ccc121da97ad70718b08a44f85f33a251c1d2baa0/yolo_preprocessing_apr13.tar.gz",
        sha256 = "8438f66f3cf810664190d83ccc121da97ad70718b08a44f85f33a251c1d2baa0",
    )

    voxel_artifact(
        name = "artifacts_03_24_2023_carry_classifier_jit",
        url = "s3://voxel-storage/artifactory/carry_object_04_18_jit/8adafca913194537801c0f059c7aa4fc020098c16fb1804eb15b26afded2cb39/carry_object_04_18_jit.tar.gz",
        sha256 = "8adafca913194537801c0f059c7aa4fc020098c16fb1804eb15b26afded2cb39",
    )

    voxel_artifact(
        name = "artifacts_04_18_2023_voxel_safetyvest_vit_quakertown_laredo_2022_08_24_jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_quakertown_laredo_2022-08-24-jit/ef0a5de3919cf5ea6d9725fb5843c47ffa09e3c4fd1772ea7389864b681746e4/voxel_safetyvest_vit_quakertown_laredo_2022-08-24-jit.tar.gz",
        sha256 = "ef0a5de3919cf5ea6d9725fb5843c47ffa09e3c4fd1772ea7389864b681746e4",
    )

    voxel_artifact(
        name = "artifacts_04_18_2023_voxel_safetyvest_vit_arlington_2022-09-08-jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_arlington_2022-09-08-jit/126489b4f76de59d0380df922efa343164052af9cd8d3ab908f419fd9791e030/voxel_safetyvest_vit_arlington_2022-09-08-jit.tar.gz",
        sha256 = "126489b4f76de59d0380df922efa343164052af9cd8d3ab908f419fd9791e030",
    )

    voxel_artifact(
        name = "artifacts_04_18_2023_voxel_safetyvest_vit_laredo-2_laredo_2022-09-15-jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_laredo-2_laredo_2022-09-15-jit/6f77dbb2d7e007863165cada977115335dea700532596f5fe362ba865e29e98e/voxel_safetyvest_vit_laredo-2_laredo_2022-09-15-jit.tar.gz",
        sha256 = "6f77dbb2d7e007863165cada977115335dea700532596f5fe362ba865e29e98e",
    )

    voxel_artifact(
        name = "artifacts_04_18_2023_voxel_safetyvest_vit_laredo_walton_2022-10-05-jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_laredo_walton_2022-10-05-jit/5862b5e8dccdc1830e0ae39e61a0b33741f6bbeceae530a83547993cbddaa14e/voxel_safetyvest_vit_laredo_walton_2022-10-05-jit.tar.gz",
        sha256 = "5862b5e8dccdc1830e0ae39e61a0b33741f6bbeceae530a83547993cbddaa14e",
    )

    voxel_artifact(
        name = "artifacts_04_18_2023_voxel_safetyvest_vit_general_2022-09-21-jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_general_2022-09-21-jit/e7e0b4c5d42d7da1b7fadc05c73ba498132a3269dec535fe3e1f2de94f3b6b08/voxel_safetyvest_vit_general_2022-09-21-jit.tar.gz",
        sha256 = "e7e0b4c5d42d7da1b7fadc05c73ba498132a3269dec535fe3e1f2de94f3b6b08",
    )

    voxel_artifact(
        name = "artifacts_04_19_2023_voxel_safetyvest_vit_dixieline_2022-09-16-jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_dixieline_2022-09-16-jit/e0a8aa6f1fbf8cf2fab4714dd43fa0db3986b0ad95c271316a0bac1469efb9da/voxel_safetyvest_vit_dixieline_2022-09-16-jit.tar.gz",
        sha256 = "e0a8aa6f1fbf8cf2fab4714dd43fa0db3986b0ad95c271316a0bac1469efb9da",
    )

    voxel_artifact(
        name = "artifacts_04_19_2023_voxel_safetyvest_vit_2022-08-30_solana_beach-jit",
        url = "s3://voxel-storage/artifactory/voxel_safetyvest_vit_2022-08-30_solana_beach-jit/9a18f6b04a2a95a5d595ba78f27aa44c0692f846f333d1b791a1d78eb980d720/voxel_safetyvest_vit_2022-08-30_solana_beach-jit.tar.gz",
        sha256 = "9a18f6b04a2a95a5d595ba78f27aa44c0692f846f333d1b791a1d78eb980d720",
    )

    voxel_artifact(
        name = "artifacts_hard_hat_occlusion_vit_v4_2022-11-18-jit",
        url = "s3://voxel-storage/artifactory/hard_hat_occlusion_vit_v4_2022-11-18-jit/168ff4ecd09a943ecb0bc47714867365a0c58d5174d2dff5e519764112cd8ace/hard_hat_occlusion_vit_v4_2022-11-18-jit.tar.gz",
        sha256 = "168ff4ecd09a943ecb0bc47714867365a0c58d5174d2dff5e519764112cd8ace",
    )

    voxel_artifact(
        name = "artifacts_bumpcap_vit_99_46_34k_2022-12-20-jit",
        url = "s3://voxel-storage/artifactory/bumpcap_vit_99_46_34k_2022-12-20-jit/b16802a14b6b9583efa81a2b1e3b3a13675bd975c6f8a3ace64a125da0eb6c0b/bumpcap_vit_99_46_34k_2022-12-20-jit.tar.gz",
        sha256 = "b16802a14b6b9583efa81a2b1e3b3a13675bd975c6f8a3ace64a125da0eb6c0b",
    )

    # Vivek: americold/ontario/0104/cha - 2023/05/02
    voxel_artifact(
        name = "artifacts_05_02_2023_americold_ontario_0104_cha",
        url = "s3://voxel-storage/artifactory/7d56133d-478e-44ab-b785-67cf3e2d69e4/969a1e742283b2ed7106a4356950de11b914d218f674b0ef36b866e41c294e48/7d56133d-478e-44ab-b785-67cf3e2d69e4.tar.gz",
        sha256 = "969a1e742283b2ed7106a4356950de11b914d218f674b0ef36b866e41c294e48",
    )

    # Vivek: jfe_shoji/burlington/0010/cha - 2023/05/02
    voxel_artifact(
        name = "artifacts_05_02_2023_jfe_shoji_burlington_0010_cha",
        url = "s3://voxel-storage/artifactory/552c8b65-55bc-4d91-937c-57686f21f67b/bc55e55c3ad45efd05aaf8f263d9e909e069756ced35e4b17b0ea6dd0f6a73e4/552c8b65-55bc-4d91-937c-57686f21f67b.tar.gz",
        sha256 = "bc55e55c3ad45efd05aaf8f263d9e909e069756ced35e4b17b0ea6dd0f6a73e4",
    )

    # Vivek: vertical_cold_storage/bolingbrook/0005/cha - 2023/05/10
    voxel_artifact(
        name = "artifacts_05_10_2023_vertical_cold_storage_bolingbrook_0005_cha",
        url = "s3://voxel-storage/artifactory/2bd41dd8-a0bf-4b9d-9ecd-f9b09a1925a4/93e838a633148ef1fda724c2988010e4ca8556d54bc4076a1e24f0cdd8c69019/2bd41dd8-a0bf-4b9d-9ecd-f9b09a1925a4.tar.gz",
        sha256 = "93e838a633148ef1fda724c2988010e4ca8556d54bc4076a1e24f0cdd8c69019",
    )

    # Vivek: vertical_cold_storage/bolingbrook/0004/cha - 2023/05/10
    voxel_artifact(
        name = "artifacts_05_10_2023_vertical_cold_storage_bolingbrook_0004_cha",
        url = "s3://voxel-storage/artifactory/b1e74b5b-058b-4724-a01d-5a00ed309364/b68eadc9ae89b92a80756669f0c1a98059cd382351faa6c6a28912b76f38e6b7/b1e74b5b-058b-4724-a01d-5a00ed309364.tar.gz",
        sha256 = "b68eadc9ae89b92a80756669f0c1a98059cd382351faa6c6a28912b76f38e6b7",
    )

    # Vivek: vertical_cold_storage/bolingbrook/0003/cha - 2023/05/10
    voxel_artifact(
        name = "artifacts_05_10_2023_vertical_cold_storage_bolingbrook_0003_cha",
        url = "s3://voxel-storage/artifactory/e8c5e8b5-0b97-4d55-b3d5-475a45884051/00c744373eefaa0135149122d95123055500c0993ac983076a84023ed789606f/e8c5e8b5-0b97-4d55-b3d5-475a45884051.tar.gz",
        sha256 = "00c744373eefaa0135149122d95123055500c0993ac983076a84023ed789606f",
    )

    voxel_artifact(
        name = "artifacts_05_10_2023_uscold_syracuse_0001_cha",
        url = "s3://voxel-storage/artifactory/79658764-90fb-4bcc-95d2-903d480fd8e6/c7a8f40eac81b23bc34aba48eaf6c499b0e814f9e6d16ce22bfc634b56baa196/79658764-90fb-4bcc-95d2-903d480fd8e6.tar.gz",
        sha256 = "c7a8f40eac81b23bc34aba48eaf6c499b0e814f9e6d16ce22bfc634b56baa196",
    )

    # Vivek: Adding new jfe_shoji_burlington YOLO model
    voxel_artifact(
        name = "artifacts_05_02_2023_jfe_shoji_burlington",
        url = "s3://voxel-storage/artifactory/best_736_1280/f99b3090d688165ebfc7bc0baef6a3da111e4933ac8d1ed1e4ee979ee941364d/best_736_1280.tar.gz",
        sha256 = "f99b3090d688165ebfc7bc0baef6a3da111e4933ac8d1ed1e4ee979ee941364d",
    )

    # Vivek: vertical_cold_storage/bolingbrook/0002/cha - 2023/05/11
    voxel_artifact(
        name = "artifacts_05_11_2023_vertical_cold_storage_bolingbrook_0002_cha",
        url = "s3://voxel-storage/artifactory/8263a6f0-0d1d-4a88-9517-15eec472f955/b63cd63fca1375e056e3705695749b3863357430dc7683e624234f9cb2770286/8263a6f0-0d1d-4a88-9517-15eec472f955.tar.gz",
        sha256 = "b63cd63fca1375e056e3705695749b3863357430dc7683e624234f9cb2770286",
    )

    # Vivek: uscold_syracuse_0008_cha - 2023/05/14
    voxel_artifact(
        name = "artifacts_05_14_2023_uscold_syracuse_0008_cha",
        url = "s3://voxel-storage/artifactory/17c121e7-33d7-4c9f-92c2-0124816696f2/9dc9fda3b50b5e6aea152746d7c9cb41c5f1431551583b948df643a691fc549d/17c121e7-33d7-4c9f-92c2-0124816696f2.tar.gz",
        sha256 = "9dc9fda3b50b5e6aea152746d7c9cb41c5f1431551583b948df643a691fc549d",
    )

    # Vivek: uscold_syracuse_0006_cha - 2023/05/14
    voxel_artifact(
        name = "artifacts_05_14_2023_uscold_syracuse_0006_cha",
        url = "s3://voxel-storage/artifactory/35d5aa7e-d16d-4ffe-a231-869a9e9614d0/0ed84bb9fbbd65163f62b56eb2a390414ae0c6576015dedcca79da6da249a73c/35d5aa7e-d16d-4ffe-a231-869a9e9614d0.tar.gz",
        sha256 = "0ed84bb9fbbd65163f62b56eb2a390414ae0c6576015dedcca79da6da249a73c",
    )

    # Vivek: uscold_syracuse_0009_cha - 2023/05/14
    voxel_artifact(
        name = "artifacts_05_14_2023_uscold_syracuse_0009_cha",
        url = "s3://voxel-storage/artifactory/4c1e3213-d405-4c3e-9ac0-17bafa91099f/45cb2cac0013c7116764ca4ddcee485efd90bdd125f9c6cdd7e8d84e0b22513e/4c1e3213-d405-4c3e-9ac0-17bafa91099f.tar.gz",
        sha256 = "45cb2cac0013c7116764ca4ddcee485efd90bdd125f9c6cdd7e8d84e0b22513e",
    )

    # Vivek: vertical_cold_storage/bolingbrook/0002/cha - 2023/05/11
    voxel_artifact(
        name = "artifacts_05_15_2023_vertical_cold_storage_bolingbrook_0001_cha",
        url = "s3://voxel-storage/artifactory/d3d65233-6807-4b69-a61e-796ffa2d6b43/85add26169159602321e981535348b7f42ca97fd6800ab03dcb039538190f856/d3d65233-6807-4b69-a61e-796ffa2d6b43.tar.gz",
        sha256 = "85add26169159602321e981535348b7f42ca97fd6800ab03dcb039538190f856",
    )

    # Walk: test data for services/platform/prism/lib/clipsynth
    voxel_artifact(
        name = "artifacts_05_17_2023_clipsynth_testdata_fragments",
        url = "s3://voxel-storage/artifactory/testdata/b9c96fb65a88dc91abdb2bf2009e9d0936111f8986ab8694ed1807c005667d60/testdata.zip",
        sha256 = "b9c96fb65a88dc91abdb2bf2009e9d0936111f8986ab8694ed1807c005667d60",
    )

    # Vivek: vertical_cold_storage/richardson/0004/cha - 2023/05/18
    voxel_artifact(
        name = "artifacts_05_18_2023_vertical_cold_storage_richardson_0004_cha",
        url = "s3://voxel-storage/artifactory/3ff43bda-8105-4942-93c8-3bff16f3b986/5d8c3e43cc786b96a28ff51ccd9e80d8a6e10b46a4f02ff5abda3bc9c3b9a6a0/3ff43bda-8105-4942-93c8-3bff16f3b986.tar.gz",
        sha256 = "5d8c3e43cc786b96a28ff51ccd9e80d8a6e10b46a4f02ff5abda3bc9c3b9a6a0",
    )

    # Vivek: vertical_cold_storage/richardson/0001/cha - 2023/05/18
    voxel_artifact(
        name = "artifacts_05_18_2023_vertical_cold_storage_richardson_0001_cha",
        url = "s3://voxel-storage/artifactory/b6de2867-c559-483d-929e-e271e1f5be81/5623ff967516c1fd807c787de236aa9ff88383f4b71d0bf819c6bc2d991e1290/b6de2867-c559-483d-929e-e271e1f5be81.tar.gz",
        sha256 = "5623ff967516c1fd807c787de236aa9ff88383f4b71d0bf819c6bc2d991e1290",
    )

    # Vivek: vertical_cold_storage/richardson/0008/cha - 2023/05/18
    voxel_artifact(
        name = "artifacts_05_18_2023_vertical_cold_storage_richardson_0008_cha",
        url = "s3://voxel-storage/artifactory/cab42a2c-0fda-4697-bad7-8e59dda9b417/8a5867ce8572cc4c793fec7146fae545a209978053d13fe198e40b5d993c795c/cab42a2c-0fda-4697-bad7-8e59dda9b417.tar.gz",
        sha256 = "8a5867ce8572cc4c793fec7146fae545a209978053d13fe198e40b5d993c795c",
    )

    # Vivek: uscold/syracuse/0007/cha - 2023/05/18
    voxel_artifact(
        name = "artifacts_05_18_2023_uscold_syracuse_0007_cha",
        url = "s3://voxel-storage/artifactory/64237aac-0b04-4b1b-bd0e-5cbb851d7e20/d29e3331222aec6a71c80f48501aef3bec8752833c08c67e66987644c838a1d6/64237aac-0b04-4b1b-bd0e-5cbb851d7e20.tar.gz",
        sha256 = "d29e3331222aec6a71c80f48501aef3bec8752833c08c67e66987644c838a1d6",
    )

    # Mamrez: Floor Segmenter
    voxel_artifact(
        name = "artifacts_05_17_2023",
        url = "s3://voxel-storage/artifactory/smp-floor-070bd18d-402b-4ba8-87cb-ad7f8304ebd6-jit-cuda/fb2edd4974275abfad4c43f7fe6c2c7eaa67faa342912669ba30d0de11711589/smp-floor-070bd18d-402b-4ba8-87cb-ad7f8304ebd6-jit-cuda.tar.gz",
        sha256 = "fb2edd4974275abfad4c43f7fe6c2c7eaa67faa342912669ba30d0de11711589",
    )

    # Vivek: vertical_cold_storage/richardson/0003/cha - 2023/05/18
    voxel_artifact(
        name = "artifacts_05_19_2023_vertical_cold_storage_richardson_0003_cha",
        url = "s3://voxel-storage/artifactory/08a12aaf-0cf5-4cea-85c0-7ea0655cd1ed/f16ea2d05f50136c4b27fa6363c1222b8a09d073e5812250937b73090ad1aff1/08a12aaf-0cf5-4cea-85c0-7ea0655cd1ed.tar.gz",
        sha256 = "f16ea2d05f50136c4b27fa6363c1222b8a09d073e5812250937b73090ad1aff1",
    )

    # Vivek: vertical_cold_storage/richardson/0006/cha - 2023/05/18
    voxel_artifact(
        name = "artifacts_05_19_2023_vertical_cold_storage_richardson_0006_cha",
        url = "s3://voxel-storage/artifactory/7de2fead-3b8f-40e9-a046-770c4db12df3/0dd81b1bcfceb8e29a4149ee6a333f8883ec5ef5d3af0f7b98c4abc5ca8fd39c/7de2fead-3b8f-40e9-a046-770c4db12df3.tar.gz",
        sha256 = "0dd81b1bcfceb8e29a4149ee6a333f8883ec5ef5d3af0f7b98c4abc5ca8fd39c",
    )

    # Vivek: Adding new vertical_cold_storage/bolingbrook YOLO model
    voxel_artifact(
        name = "artifacts_05_18_2023_vertical_cold_storage_bolingbrook_yolo",
        url = "s3://voxel-storage/artifactory/best_736_1280/76f284ac8e155e1f72b2fe6d409c12e02f26c219d76320a128560bc94b27d063/best_736_1280.tar.gz",
        sha256 = "76f284ac8e155e1f72b2fe6d409c12e02f26c219d76320a128560bc94b27d063",
    )

    # Walker: Testdata for ffprobe
    voxel_artifact(
        name = "artifacts_05_19_2023_example_incident_mp4",
        url = "s3://voxel-storage/artifactory/example-incident/6082804351afbf707b841de0d9d42601b01108d8ffad6bac4e09a27157931016/example-incident.tar.gz",
        sha256 = "6082804351afbf707b841de0d9d42601b01108d8ffad6bac4e09a27157931016",
    )

    # Vivek: verst/hebron/0006/cha - 2023/05/24
    voxel_artifact(
        name = "artifacts_05_23_2023_verst_hebron_0004_cha",
        url = "s3://voxel-storage/artifactory/014b8b5a-4d20-4f33-9437-8e23a8843750/70a5c2dae846d9bf10283de9de7a7497f5d802af38f183d142d64c893acedaf3/014b8b5a-4d20-4f33-9437-8e23a8843750.tar.gz",
        sha256 = "70a5c2dae846d9bf10283de9de7a7497f5d802af38f183d142d64c893acedaf3",
    )

    voxel_artifact(
        name = "vit_pose_model_b_multi_coco_jit",
        url = "s3://voxel-storage/artifactory/vit_pose_model_b_multi_coco_jit/fdad58168b597e3f913c172e88854cf1fcb2b4c775091a65e79f6b454537b5bf/vit_pose_model_b_multi_coco_jit.tar.gz",
        sha256 = "fdad58168b597e3f913c172e88854cf1fcb2b4c775091a65e79f6b454537b5bf",
    )

    # Vivek: vertical_cold_storage/richardson/0006/cha - 2023/05/27
    voxel_artifact(
        name = "artifacts_05_27_2023_vertical_cold_storage_richardson_0006_cha",
        url = "s3://voxel-storage/artifactory/624687fc-3c3b-4909-ab36-7219414de456/45eb5b28b31c2ed9e88bc19c637d5d26703b26be8a342a73b34e97c68b66e6d9/624687fc-3c3b-4909-ab36-7219414de456.tar.gz",
        sha256 = "45eb5b28b31c2ed9e88bc19c637d5d26703b26be8a342a73b34e97c68b66e6d9",
    )

    # Vivek: vertical_cold_storage/richardson/0002/cha - 2023/05/27
    voxel_artifact(
        name = "artifacts_05_27_2023_vertical_cold_storage_richardson_0002_cha",
        url = "s3://voxel-storage/artifactory/a5927edd-dae0-408e-92ab-779d1fcdd168/9668aa7159f3f102ee4a9654b0991fb547a53da991518ae58a88083a4358ecac/a5927edd-dae0-408e-92ab-779d1fcdd168.tar.gz",
        sha256 = "9668aa7159f3f102ee4a9654b0991fb547a53da991518ae58a88083a4358ecac",
    )

    # Vivek: wn_foods/hayward/0011/cha - 2023/05/27
    voxel_artifact(
        name = "artifacts_05_27_2023_wn_foods_hayward_0011_cha",
        url = "s3://voxel-storage/artifactory/a90eff43-24c1-468c-8b0c-244ca24062bd/015c9f3713646834b9bfb6cba48c5f34e7328820e05f1a34923a87d31672d095/a90eff43-24c1-468c-8b0c-244ca24062bd.tar.gz",
        sha256 = "015c9f3713646834b9bfb6cba48c5f34e7328820e05f1a34923a87d31672d095",
    )

    # Vivek: uscold/syracuse/0010/cha - 2023/05/28
    voxel_artifact(
        name = "artifacts_05_28_2023_uscold_syracuse_0010_cha",
        url = "s3://voxel-storage/artifactory/663a5729-6520-4626-9604-60bfe36bb0a9/0620198c874ca036b720d34344cf1022d3f4e7eccd6a5f06bfbf9edad6f2abba/663a5729-6520-4626-9604-60bfe36bb0a9.tar.gz",
        sha256 = "0620198c874ca036b720d34344cf1022d3f4e7eccd6a5f06bfbf9edad6f2abba",
    )

    # Vivek: ulta/dallas/0001/cha - 2023/05/30
    voxel_artifact(
        name = "artifacts_05_30_2023_ulta_dallas_0001_cha",
        url = "s3://voxel-storage/artifactory/338f82b5-5ab7-469e-9f8e-c33e9f97b059/173571c8c8743734ed87072fb95227b991e6de16374f83750a7dba8660379cd2/338f82b5-5ab7-469e-9f8e-c33e9f97b059.tar.gz",
        sha256 = "173571c8c8743734ed87072fb95227b991e6de16374f83750a7dba8660379cd2",
    )

    # Vivek: ulta/dallas/0004/cha - 2023/05/30
    voxel_artifact(
        name = "artifacts_05_30_2023_ulta_dallas_0004_cha",
        url = "s3://voxel-storage/artifactory/28f91392-9c28-414f-bf46-66197c005014/e6ec3a5eadc2eb4f3098b991205035642943099b50f9ea9989f407e272e66bd5/28f91392-9c28-414f-bf46-66197c005014.tar.gz",
        sha256 = "e6ec3a5eadc2eb4f3098b991205035642943099b50f9ea9989f407e272e66bd5",
    )

    # Vivek: verst/hebron/0001/cha - 2023/05/30
    voxel_artifact(
        name = "artifacts_05_30_2023_verst_hebron_0001_cha",
        url = "s3://voxel-storage/artifactory/7e996f1d-abf6-47ab-b89b-c6c87455d2e0/5c96c4f37afa9f38217f9706618f8d93039acd55ec26e783f39f6cd043dcbfa1/7e996f1d-abf6-47ab-b89b-c6c87455d2e0.tar.gz",
        sha256 = "5c96c4f37afa9f38217f9706618f8d93039acd55ec26e783f39f6cd043dcbfa1",
    )

    # Vivek: Adding new vertical_cold_storage/richardson & uscold/syracuse YOLO model
    voxel_artifact(
        name = "artifacts_05_28_2023_vertical_cold_storage_richardson_uscold_syracuse_yolo",
        url = "s3://voxel-storage/artifactory/best_736_1280/0b551079e63000c1398ad570113f5c424f18f4c7a1edc717f819c8ce3c80323b/best_736_1280.tar.gz",
        sha256 = "0b551079e63000c1398ad570113f5c424f18f4c7a1edc717f819c8ce3c80323b",
    )

    # Vivek: Adding new verst/hebron YOLO model - 2023/05/31
    voxel_artifact(
        name = "artifacts_05_31_2023_verst_hebron_yolo",
        url = "s3://voxel-storage/artifactory/best_736_1280/b70b544290d201173b8aaaa838dde9b868af151dff16db683bf20f6685b381a4/best_736_1280.tar.gz",
        sha256 = "b70b544290d201173b8aaaa838dde9b868af151dff16db683bf20f6685b381a4",
    )

    # Tim: unbatched[max=1] trt model
    voxel_artifact(
        name = "vit_pose_model_b_multi_trt_v0",
        url = "s3://voxel-storage/artifactory/vit-pose-b-multi-coco-trt-v0/d7a0dc89fc0a1a2506703cb31c9269996173b781a7963cd1f2e7a7ac81f28bf3/vit-pose-b-multi-coco-trt-v0.tar.gz",
        sha256 = "d7a0dc89fc0a1a2506703cb31c9269996173b781a7963cd1f2e7a7ac81f28bf3",
    )

    # Tim: batched[max=64] trt model
    voxel_artifact(
        name = "vit_pose_model_b_multi_trt_v1",
        url = "s3://voxel-storage/artifactory/vit-pose-b-multi-coco-trt-v1/6dd93cd7926491e0104459990b73f83b0d7e426795ee762514b8c4b6913151c0/vit-pose-b-multi-coco-trt-v1.tar.gz",
        sha256 = "6dd93cd7926491e0104459990b73f83b0d7e426795ee762514b8c4b6913151c0",
    )

    # Vivek: vertical_cold_storage/richardson/0009/cha - 2023/06/01
    voxel_artifact(
        name = "artifacts_06_01_2023_vertical_cold_storage_richardson_0009_cha",
        url = "s3://voxel-storage/artifactory/78746b27-4252-49ad-b5fe-a2ecd86c5ab4/bb2d6630448be478dc2981a4fe9107703cd9159c69d88f00c0b946a6a1829793/78746b27-4252-49ad-b5fe-a2ecd86c5ab4.tar.gz",
        sha256 = "bb2d6630448be478dc2981a4fe9107703cd9159c69d88f00c0b946a6a1829793",
    )

    # Tim: batched[max=64] trt model, fp=16
    voxel_artifact(
        name = "vit_pose_model_b_multi_trt_v2",
        url = "s3://voxel-storage/artifactory/vit-pose-b-multi-coco-trt-v2/46ee79329ce0dae7d11065f0c04a040c147390a36ad41d23808a7bcde4406740/vit-pose-b-multi-coco-trt-v2.tar.gz",
        sha256 = "46ee79329ce0dae7d11065f0c04a040c147390a36ad41d23808a7bcde4406740",
    )
