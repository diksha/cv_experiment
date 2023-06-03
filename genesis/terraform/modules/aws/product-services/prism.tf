# Lambda Function
data "aws_iam_policy_document" "incident_ingest_lambda_allow_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# Lambda policy
data "aws_iam_policy_document" "incident_ingest_lambda_access_policy" {
  statement {
    sid = "AllowS3"
    effect = "Allow"
    actions = [
      "s3:*",
    ]
    resources = [
      module.fragment_archive_bucket.bucket_arn,
      "${module.fragment_archive_bucket.bucket_arn}/*",
    ]
  }

  statement {
    sid = "AllowKinesisVideoRead"
    effect = "Allow"
    actions = [
      "kinesisvideo:Get*",
      "kinesisvideo:List*",
      "kinesisvideo:Describe*",
    ]
    resources = [
      "*"
    ]
  }

  statement {
    sid = "AllowSQS"
    effect = "Allow"
    actions = [
      "sqs:DeleteMessage",
      "sqs:ReceiveMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = [
      "${aws_sqs_queue.incident_ingest.arn}"
    ]
  }
}

resource "aws_iam_policy" "incident_ingest_lambda_access_policy" {
  name = "PrismIncidentIngestLambdaPolicy"
  description = "allows incident publisher lambda function access to kinesis and s3"
  policy = data.aws_iam_policy_document.incident_ingest_lambda_access_policy.json
}

resource "aws_iam_role" "incident_ingest_lambda_allow_assume_role" {
  name = "PrismIncidentIngestLambda-${local.environment}"
  assume_role_policy = data.aws_iam_policy_document.incident_ingest_lambda_allow_assume_role.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    aws_iam_policy.incident_ingest_lambda_access_policy.arn,
  ]
}

data "archive_file" "dummy" {
  type = "zip"
  output_path = "${path.module}/lambda_function_payload.zip"

  source {
    content = "hello"
    filename = "dummy.txt"
  }
}

resource "aws_lambda_function" "incident_ingest" {
  function_name = "voxel-${local.environment}-prism-incident-ingest"
  filename = data.archive_file.dummy.output_path
  role = aws_iam_role.incident_ingest_lambda_allow_assume_role.arn
  runtime = "go1.x"
  handler = "ingest"
  timeout = 60

  environment {
    variables = {
      PRISM_FRAGMENT_ARCHIVE_BUCKET = module.fragment_archive_bucket.bucket_name
    }
  }
}

# S3 Bucket
module "fragment_archive_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = local.target_account_id
  primary_region    = local.primary_region
  bucket_name       = "voxel-${local.environment}-prism-fragment-archive"
  enable_versioning = true
  noncurrent_days   = 1
  expiration_days   = 180
}

# SNS Topic
resource "aws_sns_topic" "incident_ingest" {
  name = "voxel-${local.environment}-prism-incident-ingest.fifo"
  fifo_topic = true
  content_based_deduplication = true
}

# SQS Queue
resource "aws_sqs_queue" "incident_ingest_dlq" {
  name = "voxel-${local.environment}-prism-incident-ingest-dlq.fifo"
  fifo_queue = true
  receive_wait_time_seconds = 20
}

resource "aws_sqs_queue" "incident_ingest" {
  name = "voxel-${local.environment}-prism-incident-ingest.fifo"
  fifo_queue = true
  receive_wait_time_seconds = 20
  visibility_timeout_seconds = 60

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.incident_ingest_dlq.arn
    maxReceiveCount     = 10
  })
}

# Connect SNS and SQS
resource "aws_sns_topic_subscription" "incident_ingest_subscription" {
  protocol  = "sqs"
  topic_arn = aws_sns_topic.incident_ingest.arn
  endpoint  = aws_sqs_queue.incident_ingest.arn
}

# SNS -> SQS Policy
data "aws_iam_policy_document" "incident_ingest_sqs_access_policy" {
  statement {
    effect = "Allow"
    principals {
      type = "Service"
      identifiers = ["sns.amazonaws.com"]
    }

    actions = [ "sqs:SendMessage" ]
    resources = [aws_sqs_queue.incident_ingest.arn]
    condition {
      test = "ArnEquals"
      variable = "aws:SourceArn"
      values = [aws_sns_topic.incident_ingest.arn]
    }
  }
}

resource "aws_sqs_queue_policy" "incident_ingest_subscription" {
  queue_url = aws_sqs_queue.incident_ingest.id
  policy = data.aws_iam_policy_document.incident_ingest_sqs_access_policy.json
}


resource "aws_iam_role_policy_attachment" "incident_ingest_lambda_access_policy_attachment" {
  role = aws_iam_role.incident_ingest_lambda_allow_assume_role.name
  policy_arn = aws_iam_policy.incident_ingest_lambda_access_policy.arn
}

# SQS Trigger
resource "aws_lambda_event_source_mapping" "incident_ingest_trigger" {
  event_source_arn = aws_sqs_queue.incident_ingest.arn
  function_name    = aws_lambda_function.incident_ingest.arn
  enabled          = true
  batch_size       = 1
}


resource "aws_cloudwatch_metric_alarm" "incident_ingest_oldest_message_alarm" {
  alarm_name          = "incident_ingest_oldest_message_alarm_${local.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "ApproximateAgeOfOldestMessage"
  namespace           = "AWS/SQS"
  period              = "300"
  statistic           = "Average"
  threshold           = "10000"

  dimensions = {
    QueueName = aws_sqs_queue.incident_ingest.name
  }

  alarm_description   = "This metric monitors the age of the oldest message in the queue and sends a notification after exceeding the specified threshold."
  alarm_actions       = [var.slack_notification_sns_topic_arn]
  ok_actions          = [var.slack_notification_sns_topic_arn]
}

resource "aws_cloudwatch_metric_alarm" "incident_ingest_dlq_oldest_message_alarm" {
  alarm_name          = "incident_ingest_dlq_oldest_message_alarm_${local.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "ApproximateAgeOfOldestMessage"
  namespace           = "AWS/SQS"
  period              = "300"
  statistic           = "Average"
  threshold           = "10000"

  dimensions = {
    QueueName = aws_sqs_queue.incident_ingest_dlq.name
  }

  alarm_description   = "This metric monitors the age of the oldest message in the queue and sends a notification after exceeding the specified threshold."
  alarm_actions       = [var.slack_notification_sns_topic_arn]
  ok_actions          = [var.slack_notification_sns_topic_arn]
}
