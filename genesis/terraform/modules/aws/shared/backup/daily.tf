locals {
  name                      = "daily"
  schedule                  = "cron(0 8 ? * * *)"
  start_window_minutes      = 60
  completion_window_minutes = 180
  retention_period_days     = 30
}

resource "aws_backup_vault" "daily" {
  name = local.name
}

resource "aws_backup_plan" "daily" {
  name = local.name

  rule {
    rule_name                = local.name
    target_vault_name        = aws_backup_vault.daily.name
    schedule                 = local.schedule
    enable_continuous_backup = true
    start_window             = local.start_window_minutes
    completion_window        = local.completion_window_minutes

    lifecycle {
      delete_after = local.retention_period_days
    }
  }
}
