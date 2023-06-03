output "db_instance_password" {
  value = module.neo4j.db_instance_password
  sensitive = true
}
