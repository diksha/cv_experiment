data "aws_iam_policy_document" "galileo_policy" {
  statement {
    sid = "galileoPolicy"
    effect    = "Allow"
    resources = ["arn:aws:eks:${var.primary_region}:${var.target_account_id}:cluster/${var.cluster_name}"]

    actions = [
      "eks:AccessKubernetesApi",
      "eks:DescribeCluster",
    ]
  }
}

resource "aws_iam_policy" "galileo_policy" {
  name        = "Galileo"
  path        = "/"
  policy = data.aws_iam_policy_document.galileo_policy.json
}

data "aws_iam_policy_document" "instance_assume_role_policy" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::273352303610:role/GalileoConnect"]
    }
  }
}

resource "aws_iam_role" "galileo" {
  name               = "Galileo"
  assume_role_policy = data.aws_iam_policy_document.instance_assume_role_policy.json
  managed_policy_arns = [aws_iam_policy.galileo_policy.arn]
}


data "aws_iam_policy_document" "cluster_autoscaler" {
  statement {
    sid = "ClusterAutoscaler"

    actions = [
      "autoscaling:DescribeAutoScalingGroups",
      "autoscaling:DescribeAutoScalingInstances",
      "autoscaling:DescribeLaunchConfigurations",
      "autoscaling:DescribeScalingActivities",
      "autoscaling:SetDesiredCapacity",
      "autoscaling:TerminateInstanceInAutoScalingGroup",
      "eks:DescribeNodegroup",
    ]

    resources = [
      "*",
    ]
  }
}

resource "aws_iam_policy" "cluster_autoscaler" {
  name   = "ClusterAutoscaler"
  policy = data.aws_iam_policy_document.cluster_autoscaler.json
}
