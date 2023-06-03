output "admin_username" {
  value = "admin"
}

output "admin_password" {
  value = random_password.argo_password.result
  sensitive = true
}