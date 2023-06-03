locals {
    s3_daily_backup_tag_key = "daily-backup"
    s3_daily_backup_tag_value = "true"
}

module "backup_setup" {
  source = "../shared/backup"
}

resource "aws_backup_selection" "s3_tag_daily" {
  iam_role_arn = module.backup_setup.iam_role_arn
  name         = "s3_tag_daily"
  plan_id      = module.backup_setup.daily_plan_id
  resources    = ["arn:aws:s3:::*"]

  condition {
    string_equals {
      key   = "aws:ResourceTag/daily-backup"
      value = "true"
    }
  }
}
