-- name: FetchAllOrganizations :many
SELECT * 
FROM public.api_organization;

-- name: FetchTopLevelZones :many
SELECT *
FROM public.zones
WHERE parent_zone_id is NULL;

-- name: CreateEdge :one
INSERT INTO public.edge (
    uuid,
    mac_address,
    "name",
    "serial",
    lifecycle,
    organization_id,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, current_timestamp, current_timestamp)
RETURNING *;

-- name: CreateZone :one
INSERT INTO public.zones (
    "name",
    zone_type,
    organization_id,
    parent_zone_id,
    "key",
    timezone,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, current_timestamp, current_timestamp)
RETURNING *;

-- name: CreateOrganization :one
INSERT INTO public.api_organization (
    "name",
    "key",
    is_sandbox,
    timezone,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, current_timestamp, current_timestamp)
RETURNING *;

-- name: UpdateCameras :many
UPDATE public.camera 
    SET lifecycle = $1,
    edge_id = $2
WHERE organization_id = $3 AND zone_id = $4
RETURNING *;
