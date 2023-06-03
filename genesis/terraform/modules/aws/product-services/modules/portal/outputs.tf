# output "portal_manifest" {
#   value = try(helm_release.portal_api_service[0].manifest, "")
# }

# output "migrate_manifest" {
#   value = try(helm_release.portal_migration_job[0].manifest, "")
# }

# output "portal_state_message_manifest" {
#   value = try(helm_release.portal_api_state_message_worker_service[0].manifest, "")
# }

# output "portal_event_message_manifest" {
#   value = try(helm_release.portal_api_event_message_worker_service[0].manifest, "")
# }
