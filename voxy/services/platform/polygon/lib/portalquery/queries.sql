-- name: GetCamera :one
SELECT * FROM public.camera
WHERE id = $1 LIMIT 1;

-- name: GetCameras :many
SELECT * FROM public.camera
ORDER BY created_at;

-- name: InsertCamera :one
INSERT INTO public.camera (Id, Created_At, Updated_At, Deleted_At, Uuid, Name, Organization_Id, Zone_Id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id;

-- name: InsertZone :one
INSERT INTO public.zones (Id, Created_At, Updated_At, Deleted_At, Name, Organization_Id, Zone_type, parent_zone_id, key, timezone, active) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING id;

-- name: InsertApiOrganization :one
INSERT INTO public.api_organization (Id, Created_At, Updated_At, Name, key, deleted_at, is_sandbox, timezone) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id;
