// Code generated by sqlc. DO NOT EDIT.
// versions:
//   sqlc v1.15.0
// source: query.sql

package polygon_initial

import (
	"context"
	"database/sql"

	"github.com/google/uuid"
)

const createEdge = `-- name: CreateEdge :one
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
RETURNING uuid, id, created_at, updated_at, deleted_at, mac_address, name, serial, lifecycle, organization_id
`

type CreateEdgeParams struct {
	Uuid           uuid.UUID
	MacAddress     sql.NullString
	Name           string
	Serial         sql.NullString
	Lifecycle      string
	OrganizationID sql.NullInt64
}

func (q *Queries) CreateEdge(ctx context.Context, arg CreateEdgeParams) (Edge, error) {
	row := q.db.QueryRowContext(ctx, createEdge,
		arg.Uuid,
		arg.MacAddress,
		arg.Name,
		arg.Serial,
		arg.Lifecycle,
		arg.OrganizationID,
	)
	var i Edge
	err := row.Scan(
		&i.Uuid,
		&i.ID,
		&i.CreatedAt,
		&i.UpdatedAt,
		&i.DeletedAt,
		&i.MacAddress,
		&i.Name,
		&i.Serial,
		&i.Lifecycle,
		&i.OrganizationID,
	)
	return i, err
}

const createOrganization = `-- name: CreateOrganization :one
INSERT INTO public.api_organization (
    "name",
    "key",
    is_sandbox,
    timezone,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, current_timestamp, current_timestamp)
RETURNING id, created_at, updated_at, name, key, deleted_at, is_sandbox, timezone
`

type CreateOrganizationParams struct {
	Name      string
	Key       string
	IsSandbox bool
	Timezone  string
}

func (q *Queries) CreateOrganization(ctx context.Context, arg CreateOrganizationParams) (ApiOrganization, error) {
	row := q.db.QueryRowContext(ctx, createOrganization,
		arg.Name,
		arg.Key,
		arg.IsSandbox,
		arg.Timezone,
	)
	var i ApiOrganization
	err := row.Scan(
		&i.ID,
		&i.CreatedAt,
		&i.UpdatedAt,
		&i.Name,
		&i.Key,
		&i.DeletedAt,
		&i.IsSandbox,
		&i.Timezone,
	)
	return i, err
}

const createZone = `-- name: CreateZone :one
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
RETURNING id, created_at, updated_at, deleted_at, name, zone_type, organization_id, parent_zone_id, key, timezone
`

type CreateZoneParams struct {
	Name           string
	ZoneType       string
	OrganizationID int64
	ParentZoneID   sql.NullInt64
	Key            string
	Timezone       sql.NullString
}

func (q *Queries) CreateZone(ctx context.Context, arg CreateZoneParams) (Zone, error) {
	row := q.db.QueryRowContext(ctx, createZone,
		arg.Name,
		arg.ZoneType,
		arg.OrganizationID,
		arg.ParentZoneID,
		arg.Key,
		arg.Timezone,
	)
	var i Zone
	err := row.Scan(
		&i.ID,
		&i.CreatedAt,
		&i.UpdatedAt,
		&i.DeletedAt,
		&i.Name,
		&i.ZoneType,
		&i.OrganizationID,
		&i.ParentZoneID,
		&i.Key,
		&i.Timezone,
	)
	return i, err
}

const fetchAllOrganizations = `-- name: FetchAllOrganizations :many
SELECT id, created_at, updated_at, name, key, deleted_at, is_sandbox, timezone 
FROM public.api_organization
`

func (q *Queries) FetchAllOrganizations(ctx context.Context) ([]ApiOrganization, error) {
	rows, err := q.db.QueryContext(ctx, fetchAllOrganizations)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []ApiOrganization
	for rows.Next() {
		var i ApiOrganization
		if err := rows.Scan(
			&i.ID,
			&i.CreatedAt,
			&i.UpdatedAt,
			&i.Name,
			&i.Key,
			&i.DeletedAt,
			&i.IsSandbox,
			&i.Timezone,
		); err != nil {
			return nil, err
		}
		items = append(items, i)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

const fetchTopLevelZones = `-- name: FetchTopLevelZones :many
SELECT id, created_at, updated_at, deleted_at, name, zone_type, organization_id, parent_zone_id, key, timezone
FROM public.zones
WHERE parent_zone_id is NULL
`

func (q *Queries) FetchTopLevelZones(ctx context.Context) ([]Zone, error) {
	rows, err := q.db.QueryContext(ctx, fetchTopLevelZones)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []Zone
	for rows.Next() {
		var i Zone
		if err := rows.Scan(
			&i.ID,
			&i.CreatedAt,
			&i.UpdatedAt,
			&i.DeletedAt,
			&i.Name,
			&i.ZoneType,
			&i.OrganizationID,
			&i.ParentZoneID,
			&i.Key,
			&i.Timezone,
		); err != nil {
			return nil, err
		}
		items = append(items, i)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

const updateCameras = `-- name: UpdateCameras :many
UPDATE public.camera 
    SET lifecycle = $1,
    edge_id = $2
WHERE organization_id = $3 AND zone_id = $4
RETURNING id, created_at, updated_at, deleted_at, uuid, name, organization_id, zone_id, thumbnail_gcs_path, edge_id, lifecycle
`

type UpdateCamerasParams struct {
	Lifecycle      sql.NullString
	EdgeID         sql.NullInt64
	OrganizationID sql.NullInt64
	ZoneID         sql.NullInt64
}

func (q *Queries) UpdateCameras(ctx context.Context, arg UpdateCamerasParams) ([]Camera, error) {
	rows, err := q.db.QueryContext(ctx, updateCameras,
		arg.Lifecycle,
		arg.EdgeID,
		arg.OrganizationID,
		arg.ZoneID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []Camera
	for rows.Next() {
		var i Camera
		if err := rows.Scan(
			&i.ID,
			&i.CreatedAt,
			&i.UpdatedAt,
			&i.DeletedAt,
			&i.Uuid,
			&i.Name,
			&i.OrganizationID,
			&i.ZoneID,
			&i.ThumbnailGcsPath,
			&i.EdgeID,
			&i.Lifecycle,
		); err != nil {
			return nil, err
		}
		items = append(items, i)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}