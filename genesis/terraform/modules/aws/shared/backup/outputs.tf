output "iam_role_arn" {
    value = aws_iam_role.service_role.arn
}

output "daily_plan_id" {
    value =aws_backup_plan.daily.id
}
