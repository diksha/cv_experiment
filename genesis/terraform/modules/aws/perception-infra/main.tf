moved {
  from = module.perception_base_setup.module.vpc_primary
  to = module.vpc.module.vpc_primary
}

locals {
  perception_alias = "perception"
  portal_alias     = "portal"
}

module "vpc" {
  providers = {
    aws = aws
  }
  source             = "../shared/vpc-and-subnets"
  target_account_id  = var.target_account_id
  environment        = var.environment
  vpc_cidr_root      = var.perception_vpc_cidr_root
  vpc_name           = local.perception_alias
  enable_nat_gateway = true
  public_subnet_tags = {
    "kubernetes.io/role/elb"                          = "1"
    "kubernetes.io/cluster/${local.perception_alias}" = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"                 = "1"
    "kubernetes.io/cluster/${local.perception_alias}" = "shared"
  }
}

module "perception_base_setup" {
  providers = {
    aws = aws
  }
  source                                  = "../shared/base-env"
  account_id                              = var.target_account_id
  primary_region                          = var.primary_region
  environment                             = var.environment
  dev_mode                                = lower(var.environment) == "production" ? false : true
  eks_should_create_standard_node_group   = true
  eks_should_create_gpu_node_group        = true
  vpc_cidr_root                           = var.perception_vpc_cidr_root
  vpc_name                                = local.perception_alias
  vpc_id                                  = module.vpc.vpc_id
  private_subnets                         = module.vpc.private_subnet_ids
  security_group_ids                      = module.vpc.vpc_security_group_ids
  cluster_name                            = local.perception_alias
  eks_default_max_instance_count          = var.eks_default_max_instance_count
  eks_cpu_instance_types                  = split(",", var.perception_eks_cpu_instance_types_comma_separated, )
  eks_gpu_instance_types                  = split(",", var.perception_eks_gpu_instance_types_comma_separated)
  eks_default_disk_size                   = 200
  cluster_version                         = "1.24"
  aws_notifications_slack_app_webhook_url = var.aws_notifications_slack_app_webhook_url
  create_alarm                            = lower(var.environment) == "production" ? true : false
  eks_extra_node_groups = {
    gpu_medium_public = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["g4dn.2xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      subnet_ids     = module.vpc.public_subnet_ids
      disk_size      = 200
      block_device_mappings = [
        {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 3000
            encrypted             = true
            delete_on_termination = true
          }
        },
      ]
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      },
      {
        key    = "subnet"
        value  = "public"
        effect = "NO_SCHEDULE"
      }]
      tags = {
        "nvidia.com/gpu"                                                          = "true"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
        "k8s.io/cluster-autoscaler/node-template/label/subnet"                    = "public"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: subnet"         = "public:NoSchedule"
      }
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "true"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
        "k8s.io/cluster-autoscaler/node-template/label/subnet"                    = "public"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: subnet"         = "public:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "true"
        "subnet"         = "public"
      }
      min_size = 0
      max_size = 50
      iam_role_additional_policies = {
        kinesis_read_only = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
        ssm_access = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      }
    }
  }
}

module "perception_observability" {
  providers = {
    aws = aws
  }
  source                   = "../shared/eks-observability"
  account_id               = var.target_account_id
  cluster_name             = module.perception_base_setup.eks_cluster_name
  grafana_url              = var.grafana_url
  grafana_irsa_arn         = var.grafana_irsa_arn
  grafana_api_key          = var.grafana_api_key
  observability_identifier = "${var.observability_identifier}-${local.perception_alias}"
  register_with_grafana    = true
  aws_region               = var.primary_region
  install_gpu_components   = true
  prometheus_memory        = "10G"
}


resource "aws_vpc_endpoint" "s3_gateway_primary_region" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${var.primary_region}.s3"
  route_table_ids = concat(
    tolist(module.vpc.private_route_table_ids),
    tolist(module.vpc.public_route_table_ids),
  )
}


resource "aws_iam_openid_connect_provider" "idp_for_prime_jenkins" {
  url            = "https://oidc.eks.us-west-2.amazonaws.com/id/7F736585AF1A879653C8E0819543D12E"
  client_id_list = ["sts.amazonaws.com"]
  # Default Thumbprint of Root CA for EKS OIDC - Valid until 2037.
  thumbprint_list = ["9e99a48a9960b14926bb7f3b02e22da2b0ab7280"]
}

resource "aws_kinesis_stream" "perception-states-events" {
  name             = "perception-states-events"
  shard_count      = 4
  retention_period = 24

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
  ]

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }
}

resource "aws_kinesis_stream" "perception-frame-structs" {
  name             = "perception-frame-structs"
  shard_count      = 4
  retention_period = 24

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
  ]

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }
}

resource "aws_iam_policy" "perception_kinesis_write_only" {
  name        = "PerceptionKinesisWriteOnly"
  description = "Write only access to the perception kinesis streams"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "WriteAccess",
        Action = [
          "kinesis:PutRecord",
          "kinesis:PutRecords",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:kinesis:us-west-2:${var.target_account_id}:stream/${aws_kinesis_stream.perception-frame-structs.name}",
          "arn:aws:kinesis:us-west-2:${var.target_account_id}:stream/${aws_kinesis_stream.perception-states-events.name}",
        ],
      },
    ],
  })
}

resource "aws_iam_role" "lambda_execution" {
  name = "LambdaExecutionRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
  managed_policy_arns = ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"]
}

data "archive_file" "dummy" {
  type        = "zip"
  output_path = "${path.module}/lambda_function_payload.zip"

  source {
    content  = "hello"
    filename = "dummy.txt"
  }
}

resource "aws_lambda_function" "kds_extract_partition_key" {
  function_name = "voxel-kds-extract-partition-key"
  filename      = data.archive_file.dummy.output_path
  role          = aws_iam_role.lambda_execution.arn
  runtime       = "go1.x"
  handler       = "kds-extract-camera-uuid"
  timeout       = 30
}

resource "aws_lambda_function" "perception_frame_structs_firehose_transform" {
  function_name = "voxel-perception-frame-structs-firehose-transform"
  filename      = data.archive_file.dummy.output_path
  role          = aws_iam_role.lambda_execution.arn
  runtime       = "go1.x"
  handler       = "kds-extract-camera-uuid"
  timeout       = 30
}

module "frame_structs_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-perception-${var.environment}-frame-structs"
  enable_versioning = true
  noncurrent_days   = 30
  expiration_days   = 180
}

module "states_events_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-perception-${var.environment}-states-events"
  enable_versioning = true
  noncurrent_days   = 30
  expiration_days   = 180
}

data "aws_iam_policy_document" "firehose_assume_role_policy" {
  statement {
    sid     = "allowfirehoseassumerole"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["firehose.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "firehose_access_policy" {
  statement {
    sid = "allows3"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetBucketLocation",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:ListBucketMultipartUploads",
      "s3:PutObject"
    ]
    resources = [
      "arn:aws:s3:::${module.states_events_bucket.bucket_name}",
      "arn:aws:s3:::${module.states_events_bucket.bucket_name}/*",
      "arn:aws:s3:::${module.frame_structs_bucket.bucket_name}",
      "arn:aws:s3:::${module.frame_structs_bucket.bucket_name}/*",
    ]
  }

  statement {
    sid = "allowkinesispull"
    actions = [
      "kinesis:DescribeStream",
      "kinesis:GetShardIterator",
      "kinesis:GetRecords",
      "kinesis:ListShards"
    ]
    resources = [
      aws_kinesis_stream.perception-states-events.arn,
      aws_kinesis_stream.perception-frame-structs.arn
    ]
  }

  statement {
    sid = "allowlambda"

    actions = [
      "lambda:InvokeFunction",
      "lambda:GetFunctionConfiguration"
    ]

    resources = [
      aws_lambda_function.kds_extract_partition_key.arn,
      aws_lambda_function.perception_frame_structs_firehose_transform.arn
    ]
  }
}

resource "aws_iam_policy" "firehose_access_policy" {
  name        = "perception-firehose-access-policy"
  description = "allows a firehose delivery stream access to kinesis and s3"
  policy      = data.aws_iam_policy_document.firehose_access_policy.json
}

resource "aws_iam_role" "firehose_role" {
  name               = "FirehoseAssumeRole"
  assume_role_policy = data.aws_iam_policy_document.firehose_assume_role_policy.json
}

resource "aws_iam_role_policy_attachment" "firehose_role_access_policy_attachment" {
  role       = aws_iam_role.firehose_role.name
  policy_arn = aws_iam_policy.firehose_access_policy.arn
}

resource "aws_kinesis_firehose_delivery_stream" "perception_states_events" {
  name        = "perception-states-events"
  destination = "extended_s3"

  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.perception-states-events.arn
    role_arn           = aws_iam_role.firehose_role.arn
  }

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose_role.arn
    bucket_arn = module.states_events_bucket.bucket_arn

    dynamic_partitioning_configuration {
      enabled = "true"
    }

    # Example prefix using partitionKeyFromQuery, applicable to JQ processor
    prefix              = "data/!{partitionKeyFromLambda:camera_uuid}/!{timestamp:yyyy/MM/dd}/"
    error_output_prefix = "errors/!{timestamp:yyyy/MM/dd}/!{firehose:error-output-type}/"

    # https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning.html
    buffer_size     = 64
    buffer_interval = 900

    processing_configuration {
      enabled = "true"
      processors {
        type = "Lambda"
        parameters {
          parameter_name  = "LambdaArn"
          parameter_value = aws_lambda_function.kds_extract_partition_key.arn
        }

        parameters {
          parameter_name  = "BufferSizeInMBs"
          parameter_value = "2"
        }

        parameters {
          parameter_name  = "BufferIntervalInSeconds"
          parameter_value = "120"
        }
      }

      processors {
        type = "AppendDelimiterToRecord"
      }
    }
  }
}

resource "aws_kinesis_firehose_delivery_stream" "perception_frame_structs" {
  name        = "perception-frame-structs"
  destination = "extended_s3"

  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.perception-frame-structs.arn
    role_arn           = aws_iam_role.firehose_role.arn
  }

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose_role.arn
    bucket_arn = module.frame_structs_bucket.bucket_arn

    dynamic_partitioning_configuration {
      enabled = "true"
    }

    # Example prefix using partitionKeyFromQuery, applicable to JQ processor
    prefix              = "data/!{partitionKeyFromLambda:camera_uuid}/!{timestamp:yyyy/MM/dd}/"
    error_output_prefix = "errors/!{timestamp:yyyy/MM/dd}/!{firehose:error-output-type}/"

    # https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning.html
    buffer_size     = 64
    buffer_interval = 900

    processing_configuration {
      enabled = "true"
      processors {
        type = "Lambda"
        parameters {
          parameter_name  = "LambdaArn"
          parameter_value = aws_lambda_function.perception_frame_structs_firehose_transform.arn
        }

        parameters {
          parameter_name  = "BufferSizeInMBs"
          parameter_value = "2"
        }

        parameters {
          parameter_name  = "BufferIntervalInSeconds"
          parameter_value = "120"
        }
      }

      processors {
        type = "AppendDelimiterToRecord"
      }
    }
  }
}

data "aws_iam_policy_document" "developer_access_read_only_policy" {
  statement {
    sid = "readonly"
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"]
    }

    actions = [
      "s3:ListBucketMultipartUploads",
      "s3:ListBucketVersions",
      "s3:ListBucket",
      "s3:ListMultipartUploadParts",
      "s3:GetObjectAcl",
      "s3:GetObject",
      "s3:GetBucketAcl",
      "s3:GetObjectVersionAcl",
      "s3:GetObjectVersion",
      "s3:GetBucketLocation"
    ]
    resources = [
      module.states_events_bucket.bucket_arn,
      "${module.states_events_bucket.bucket_arn}/*",

    ]
  }
}

resource "aws_s3_bucket_policy" "frame_structs_bucket_policy" {
  bucket = module.frame_structs_bucket.bucket_id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "readonly"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation"
        ]
        Principal = {
          AWS = "arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"
        }
        Effect = "Allow"
        Resource = [
          module.frame_structs_bucket.bucket_arn,
          "${module.frame_structs_bucket.bucket_arn}/*"
        ]
      },
    ]
  })
}

resource "aws_s3_bucket_policy" "states_events_bucket_policy" {
  bucket = module.states_events_bucket.bucket_id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "readonly"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation"
        ]
        Principal = {
          AWS = "arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"
        }
        Effect = "Allow"
        Resource = [
          module.states_events_bucket.bucket_arn,
          "${module.states_events_bucket.bucket_arn}/*"
        ]
      },
    ]
  })
}
