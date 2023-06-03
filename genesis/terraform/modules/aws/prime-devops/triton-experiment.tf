resource "kubernetes_namespace_v1" "triton-experiment" {
    metadata {
        name = "triton-experiment"
    }
}

module "triton-experiment" {
    source = "../shared/triton-service"
    name = "triton-experiment-service"
    namespace = "triton-experiment"
    oidc_provider = local.oidc_provider
    create_namespace = false
    model_repository = "s3://voxel-experimental-triton-models/ensemble-experiment"
    model_control_mode = "poll"
}

module "triton-experiment-norepo" {
    source = "../shared/triton-service"
    name = "triton-experiment-service-norepo"
    namespace = "triton-experiment"
    oidc_provider = local.oidc_provider
    create_namespace = false
    model_repository = "/tmp"
    model_control_mode = "explicit"
}

module "triton-experimental-models-bucket" {
    source = "../shared/s3-bucket"
    target_account_id          = var.target_account_id
    primary_region             = var.primary_region
    bucket_name                = "voxel-experimental-triton-models"
    enable_versioning          = true
    noncurrent_days            = 90
}
