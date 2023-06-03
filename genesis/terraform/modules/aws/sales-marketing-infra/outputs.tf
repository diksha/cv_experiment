output "db_instance_endpoint" {
  value = module.sales_marketing_db.db_instance_endpoint
}
output "db_instance_name" {
  value = module.sales_marketing_db.db_instance_name
}
output "db_instance_username" {
  value     = module.sales_marketing_db.db_instance_username
  sensitive = true
}
output "db_instance_password" {
  value     = module.sales_marketing_db.db_instance_password
  sensitive = true
}

output "db_instance_address" {
  value = module.sales_marketing_db.db_instance_address
}

output "db_instance_port" {
  value = module.sales_marketing_db.db_instance_port
}
