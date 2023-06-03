output "production_account_id" {
  value = module.production_account_init.account_id
}

output "staging_account_id" {
  value = module.staging_account_init.account_id
}

output "development_account_id" {
  value = module.development_account_init.account_id
}

output "platform_account_id" {
  value = module.platform_account_init.account_id
}

output "prime_account_id" {
  value = module.prime_account_init.account_id
}

output "galileo_account_id" {
  value = module.galileo_account_init.account_id
}

output "root_account_id" {
  value = var.root_account_id
}