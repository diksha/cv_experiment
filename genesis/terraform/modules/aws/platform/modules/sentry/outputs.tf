output "admin_username" {
  value = "admin"
}

output "admin_password" {
  value = random_password.admin_password.result
  sensitive = true
}