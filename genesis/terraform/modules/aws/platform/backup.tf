module "backup_setup" {
  providers = {
    aws = aws
  }
  source = "../shared/backup"
}

resource "aws_backup_selection" "all_ebs_daily" {
  iam_role_arn = module.backup_setup.iam_role_arn
  name         = "all_ebs_daily"
  plan_id      = module.backup_setup.daily_plan_id
  resources    = ["arn:aws:ec2:*:*:volume/*"]
}
