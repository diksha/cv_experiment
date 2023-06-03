resource "aws_ecr_repository" "lambda_noop_image" {
    name = "third_party/aws/lambda/noop"

    image_scanning_configuration {
        scan_on_push = true
    }
}

data "aws_iam_policy_document" "lambda_noop_image" {
    statement {
        sid = "ReadOnly"
        actions = [
            "ecr:BatchGetImage",
            "ecr:GetDownloadUrlForLayer",
        ]
        principals {
            type = "AWS"
            # grant read access to all of our accounts
            identifiers = [for name, props in var.context.accounts: "arn:aws:iam::${props.id}:root"]
        }
    }
    statement {
        sid = "CrossAccount"
        actions = [
            "ecr:BatchGetImage",
            "ecr:GetDownloadUrlForLayer",
        ]

        principals {
            type = "Service"
            identifiers = ["lambda.amazonaws.com"]
        }

        condition {
            test = "ForAnyValue:StringLike"
            variable = "aws:sourceArn"
            values = [for name, props in var.context.accounts: "arn:aws:lambda:*:${props.id}:function:*"]
        }
    }
}

resource "aws_ecr_repository_policy" "lambda_noop_image" {
    repository = aws_ecr_repository.lambda_noop_image.name
    policy = data.aws_iam_policy_document.lambda_noop_image.json
}
