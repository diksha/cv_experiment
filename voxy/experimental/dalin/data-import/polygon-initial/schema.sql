-- public.api_incidenttype definition

-- Drop table

-- DROP TABLE api_incidenttype;

CREATE TABLE api_incidenttype (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	value varchar(100) NOT NULL,
	background_color varchar(7) NOT NULL,
	"key" varchar(100) NOT NULL,
	"name" varchar(100) NOT NULL,
	category varchar(25) NULL,
	CONSTRAINT api_incidenttype_key_key UNIQUE (key),
	CONSTRAINT api_incidenttype_name_key UNIQUE (name),
	CONSTRAINT api_incidenttype_pkey PRIMARY KEY (id),
	CONSTRAINT api_incidenttype_value_key UNIQUE (value)
);
CREATE INDEX api_incidenttype_key_d0b1ecab_like ON public.api_incidenttype USING btree (key varchar_pattern_ops);
CREATE INDEX api_incidenttype_name_8c18305b_like ON public.api_incidenttype USING btree (name varchar_pattern_ops);
CREATE INDEX api_incidenttype_value_34d38339_like ON public.api_incidenttype USING btree (value varchar_pattern_ops);


-- public.api_organization definition

-- Drop table

-- DROP TABLE api_organization;

CREATE TABLE api_organization (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	"name" varchar(250) NOT NULL,
	"key" varchar(50) NOT NULL,
	deleted_at timestamptz NULL,
	is_sandbox bool NOT NULL,
	timezone varchar(50) NOT NULL,
	CONSTRAINT api_organization_key_key UNIQUE (key),
	CONSTRAINT api_organization_pkey PRIMARY KEY (id)
);
CREATE INDEX api_organization_key_3cd7f618_like ON public.api_organization USING btree (key varchar_pattern_ops);


-- public.auth_group definition

-- Drop table

-- DROP TABLE auth_group;

CREATE TABLE auth_group (
	id serial4 NOT NULL,
	"name" varchar(150) NOT NULL,
	CONSTRAINT auth_group_name_key UNIQUE (name),
	CONSTRAINT auth_group_pkey PRIMARY KEY (id)
);
CREATE INDEX auth_group_name_a6ea08ec_like ON public.auth_group USING btree (name varchar_pattern_ops);


-- public.auth_user definition

-- Drop table

-- DROP TABLE auth_user;

CREATE TABLE auth_user (
	id serial4 NOT NULL,
	"password" varchar(128) NOT NULL,
	last_login timestamptz NULL,
	is_superuser bool NOT NULL,
	username varchar(150) NOT NULL,
	first_name varchar(150) NOT NULL,
	last_name varchar(150) NOT NULL,
	email varchar(254) NOT NULL,
	is_staff bool NOT NULL,
	is_active bool NOT NULL,
	date_joined timestamptz NOT NULL,
	CONSTRAINT auth_user_pkey PRIMARY KEY (id),
	CONSTRAINT auth_user_username_key UNIQUE (username)
);
CREATE INDEX auth_user_username_6821ab7c_like ON public.auth_user USING btree (username varchar_pattern_ops);


-- public.camera_lifecycle definition

-- Drop table

-- DROP TABLE camera_lifecycle;

CREATE TABLE camera_lifecycle (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"key" varchar(64) NOT NULL,
	description varchar(256) NOT NULL,
	CONSTRAINT camera_lifecycle_key_key UNIQUE (key),
	CONSTRAINT camera_lifecycle_pkey PRIMARY KEY (id)
);
CREATE INDEX camera_lifecycle_key_5d819764_like ON public.camera_lifecycle USING btree (key varchar_pattern_ops);


-- public.compliance_type definition

-- Drop table

-- DROP TABLE compliance_type;

CREATE TABLE compliance_type (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"key" varchar(100) NOT NULL,
	"name" varchar(250) NOT NULL,
	CONSTRAINT compliance_type_key_ab5c2118_uniq UNIQUE (key),
	CONSTRAINT compliance_type_pkey PRIMARY KEY (id)
);
CREATE INDEX compliance_type_key_ab5c2118_like ON public.compliance_type USING btree (key varchar_pattern_ops);


-- public.edge_lifecycle definition

-- Drop table

-- DROP TABLE edge_lifecycle;

CREATE TABLE edge_lifecycle (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"key" varchar(64) NOT NULL,
	description varchar(256) NOT NULL,
	CONSTRAINT edge_lifecycle_key_key UNIQUE (key),
	CONSTRAINT edge_lifecycle_pkey PRIMARY KEY (id)
);
CREATE INDEX edge_lifecycle_key_3400bbeb_like ON public.edge_lifecycle USING btree (key varchar_pattern_ops);


-- public.api_list definition

-- Drop table

-- DROP TABLE api_list;

CREATE TABLE api_list (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	"name" varchar(250) NOT NULL,
	is_starred_list bool NOT NULL,
	owner_id int4 NOT NULL,
	deleted_at timestamptz NULL,
	CONSTRAINT api_list_owner_id_key UNIQUE (owner_id),
	CONSTRAINT api_list_pkey PRIMARY KEY (id),
	CONSTRAINT api_list_owner_id_945bb8c8_fk_auth_user_id FOREIGN KEY (owner_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);


-- public.api_organization_users definition

-- Drop table

-- DROP TABLE api_organization_users;

CREATE TABLE api_organization_users (
	id bigserial NOT NULL,
	organization_id int8 NOT NULL,
	user_id int4 NOT NULL,
	CONSTRAINT api_organization_users_organization_id_user_id_0c2fe35f_uniq UNIQUE (organization_id, user_id),
	CONSTRAINT api_organization_users_pkey PRIMARY KEY (id),
	CONSTRAINT api_organization_users_organization_id_a538c02c_fk FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_organization_users_user_id_d85b1b69_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_organization_users_organization_id_a538c02c ON public.api_organization_users USING btree (organization_id);
CREATE INDEX api_organization_users_user_id_d85b1b69 ON public.api_organization_users USING btree (user_id);


-- public.api_organizationincidenttype definition

-- Drop table

-- DROP TABLE api_organizationincidenttype;

CREATE TABLE api_organizationincidenttype (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	enabled bool NOT NULL,
	incident_type_id int8 NOT NULL,
	organization_id int8 NOT NULL,
	name_override varchar(100) NULL,
	review_level varchar(20) NOT NULL,
	CONSTRAINT api_organizationincident_incident_type_id_organiz_91b27fde_uniq UNIQUE (incident_type_id, organization_id),
	CONSTRAINT api_organizationincidenttype_pkey PRIMARY KEY (id),
	CONSTRAINT api_organizationinci_incident_type_id_feb3f07a_fk_api_incid FOREIGN KEY (incident_type_id) REFERENCES api_incidenttype(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_organizationinci_organization_id_3379cd0d_fk_api_organ FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_organizationincidenttype_incident_type_id_feb3f07a ON public.api_organizationincidenttype USING btree (incident_type_id);
CREATE INDEX api_organizationincidenttype_organization_id_3379cd0d ON public.api_organizationincidenttype USING btree (organization_id);


-- public.auth_user_groups definition

-- Drop table

-- DROP TABLE auth_user_groups;

CREATE TABLE auth_user_groups (
	id bigserial NOT NULL,
	user_id int4 NOT NULL,
	group_id int4 NOT NULL,
	CONSTRAINT auth_user_groups_pkey PRIMARY KEY (id),
	CONSTRAINT auth_user_groups_user_id_group_id_94350c0c_uniq UNIQUE (user_id, group_id),
	CONSTRAINT auth_user_groups_group_id_97559544_fk_auth_group_id FOREIGN KEY (group_id) REFERENCES auth_group(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT auth_user_groups_user_id_6a12ed8b_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX auth_user_groups_group_id_97559544 ON public.auth_user_groups USING btree (group_id);
CREATE INDEX auth_user_groups_user_id_6a12ed8b ON public.auth_user_groups USING btree (user_id);


-- public.authtoken_token definition

-- Drop table

-- DROP TABLE authtoken_token;

CREATE TABLE authtoken_token (
	"key" varchar(40) NOT NULL,
	created timestamptz NOT NULL,
	user_id int4 NOT NULL,
	CONSTRAINT authtoken_token_pkey PRIMARY KEY (key),
	CONSTRAINT authtoken_token_user_id_key UNIQUE (user_id),
	CONSTRAINT authtoken_token_user_id_35299eff_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX authtoken_token_key_10f0b77e_like ON public.authtoken_token USING btree (key varchar_pattern_ops);


-- public.edge definition

-- Drop table

-- DROP TABLE edge;

CREATE TABLE edge (
	uuid uuid NOT NULL,
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	mac_address varchar(256) NULL,
	"name" varchar(256) NOT NULL,
	serial varchar(64) NULL,
	lifecycle varchar(64) NOT NULL,
	organization_id int8 NULL,
	CONSTRAINT edge_pkey PRIMARY KEY (id),
	CONSTRAINT edge_uuid_key UNIQUE (uuid),
	CONSTRAINT edge_lifecycle_81ea466b_fk_edge_lifecycle_key FOREIGN KEY (lifecycle) REFERENCES edge_lifecycle("key") DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT edge_organization_id_4a2f88eb_fk_api_organization_id FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX edge_lifecycle_81ea466b ON public.edge USING btree (lifecycle);
CREATE INDEX edge_lifecycle_81ea466b_like ON public.edge USING btree (lifecycle varchar_pattern_ops);
CREATE INDEX edge_organiz_c9ab5b_idx ON public.edge USING btree (organization_id);
CREATE INDEX edge_organization_id_4a2f88eb ON public.edge USING btree (organization_id);


-- public.zones definition

-- Drop table

-- DROP TABLE zones;

CREATE TABLE zones (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"name" varchar(250) NOT NULL,
	zone_type varchar(10) NOT NULL,
	organization_id int8 NOT NULL,
	parent_zone_id int8 NULL,
	"key" varchar(50) NOT NULL,
	timezone varchar(50) NULL,
	CONSTRAINT unique_key_per_org_and_parent UNIQUE (organization_id, parent_zone_id, key),
	CONSTRAINT zones_key_63090387_uniq UNIQUE (key),
	CONSTRAINT zones_pkey PRIMARY KEY (id),
	CONSTRAINT zones_organization_id_834e7b3e_fk_api_organization_id FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT zones_parent_zone_id_8de173dd_fk_zones_id FOREIGN KEY (parent_zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX zones_key_63090387_like ON public.zones USING btree (key varchar_pattern_ops);
CREATE INDEX zones_organization_id_834e7b3e ON public.zones USING btree (organization_id);
CREATE INDEX zones_parent_zone_id_8de173dd ON public.zones USING btree (parent_zone_id);


-- public.zones_zoneuser definition

-- Drop table

-- DROP TABLE zones_zoneuser;

CREATE TABLE zones_zoneuser (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	user_id int4 NOT NULL,
	zone_id int8 NOT NULL,
	is_assignable bool NOT NULL,
	CONSTRAINT zones_zoneuser_pkey PRIMARY KEY (id),
	CONSTRAINT zones_zoneuser_zone_id_user_id_9a576deb_uniq UNIQUE (zone_id, user_id),
	CONSTRAINT zones_zoneuser_user_id_51873474_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT zones_zoneuser_zone_id_e4e811a9_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX zones_zoneuser_user_id_51873474 ON public.zones_zoneuser USING btree (user_id);
CREATE INDEX zones_zoneuser_zone_id_e4e811a9 ON public.zones_zoneuser USING btree (zone_id);


-- public.api_profile definition

-- Drop table

-- DROP TABLE api_profile;

CREATE TABLE api_profile (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	owner_id int4 NOT NULL,
	"data" jsonb NULL,
	organization_id int8 NULL,
	deleted_at timestamptz NULL,
	timezone varchar(50) NOT NULL,
	site_id int8 NULL,
	CONSTRAINT api_profile_owner_id_key UNIQUE (owner_id),
	CONSTRAINT api_profile_pkey PRIMARY KEY (id),
	CONSTRAINT api_profile_organization_id_410907d7_fk FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_profile_owner_id_bb82c117_fk_auth_user_id FOREIGN KEY (owner_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_profile_site_id_e9a6c383_fk_zones_id FOREIGN KEY (site_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_profile_organization_id_410907d7 ON public.api_profile USING btree (organization_id);
CREATE INDEX api_profile_site_id_e9a6c383 ON public.api_profile USING btree (site_id);


-- public.camera definition

-- Drop table

-- DROP TABLE camera;

CREATE TABLE camera (
	id int8 NOT NULL DEFAULT nextval('cameras_id_seq'::regclass),
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	uuid varchar(250) NOT NULL,
	"name" varchar(250) NOT NULL,
	organization_id int8 NULL,
	zone_id int8 NULL,
	thumbnail_gcs_path varchar(250) NULL,
	edge_id int8 NULL,
	lifecycle varchar(64) NULL,
	CONSTRAINT cameras_pkey PRIMARY KEY (id),
	CONSTRAINT cameras_uuid_6b6f17d1_uniq UNIQUE (uuid),
	CONSTRAINT camera_edge_id_21ed6ee2_fk_edge_id FOREIGN KEY (edge_id) REFERENCES edge(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT camera_lifecycle_7c211357_fk_camera_lifecycle_key FOREIGN KEY (lifecycle) REFERENCES camera_lifecycle("key") DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT camera_zone_id_d365139a_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT cameras_organization_id_8f6e273a_fk_api_organization_id FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX camera_edge_id_21ed6ee2 ON public.camera USING btree (edge_id);
CREATE INDEX camera_lifecycle_7c211357 ON public.camera USING btree (lifecycle);
CREATE INDEX camera_lifecycle_7c211357_like ON public.camera USING btree (lifecycle varchar_pattern_ops);
CREATE INDEX camera_organiz_6af57f_idx ON public.camera USING btree (organization_id);
CREATE INDEX camera_zone_id_b8610a_idx ON public.camera USING btree (zone_id);
CREATE INDEX camera_zone_id_d365139a ON public.camera USING btree (zone_id);
CREATE INDEX cameras_organization_id_8f6e273a ON public.camera USING btree (organization_id);
CREATE INDEX cameras_uuid_6b6f17d1_like ON public.camera USING btree (uuid varchar_pattern_ops);


-- public.cameraconfignew definition

-- Drop table

-- DROP TABLE cameraconfignew;

CREATE TABLE cameraconfignew (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	doors jsonb NULL,
	driving_areas jsonb NULL,
	actionable_regions jsonb NULL,
	intersections jsonb NULL,
	end_of_aisles jsonb NULL,
	no_pedestrian_zones jsonb NULL,
	"version" int4 NOT NULL,
	camera_id int8 NOT NULL,
	motion_detection_zones jsonb NULL,
	CONSTRAINT cameraconfignew_camera_id_version_231b0145_uniq UNIQUE (camera_id, version),
	CONSTRAINT cameraconfignew_pkey PRIMARY KEY (id),
	CONSTRAINT cameraconfignew_camera_id_cad68d12_fk_camera_id FOREIGN KEY (camera_id) REFERENCES camera(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX cameraconfignew_camera_id_cad68d12 ON public.cameraconfignew USING btree (camera_id);


-- public.door_event_aggregates definition

-- Drop table

-- DROP TABLE door_event_aggregates;

CREATE TABLE door_event_aggregates (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	group_key timestamptz NOT NULL,
	group_by varchar(25) NOT NULL,
	max_timestamp timestamptz NOT NULL,
	opened_count int4 NOT NULL,
	closed_within_30_seconds_count int4 NOT NULL,
	closed_within_1_minute_count int4 NOT NULL,
	closed_within_5_minutes_count int4 NOT NULL,
	closed_within_10_minutes_count int4 NOT NULL,
	camera_id int8 NOT NULL,
	organization_id int8 NOT NULL,
	zone_id int8 NOT NULL,
	CONSTRAINT door_event_aggregates_closed_within_10_minutes_count_check CHECK ((closed_within_10_minutes_count >= 0)),
	CONSTRAINT door_event_aggregates_closed_within_1_minute_count_check CHECK ((closed_within_1_minute_count >= 0)),
	CONSTRAINT door_event_aggregates_closed_within_30_seconds_count_check CHECK ((closed_within_30_seconds_count >= 0)),
	CONSTRAINT door_event_aggregates_closed_within_5_minutes_count_check CHECK ((closed_within_5_minutes_count >= 0)),
	CONSTRAINT door_event_aggregates_opened_count_check CHECK ((opened_count >= 0)),
	CONSTRAINT door_event_aggregates_pkey PRIMARY KEY (id),
	CONSTRAINT unique_row_per_camera_per_group_by_option UNIQUE (group_key, group_by, organization_id, zone_id, camera_id),
	CONSTRAINT door_event_aggregate_organization_id_27f8e792_fk_api_organ FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT door_event_aggregates_camera_id_921d2781_fk_camera_id FOREIGN KEY (camera_id) REFERENCES camera(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT door_event_aggregates_zone_id_88465020_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX door_event__group_k_272b28_idx ON public.door_event_aggregates USING btree (group_key DESC, group_by, organization_id, zone_id);
CREATE INDEX door_event_aggregates_camera_id_921d2781 ON public.door_event_aggregates USING btree (camera_id);
CREATE INDEX door_event_aggregates_organization_id_27f8e792 ON public.door_event_aggregates USING btree (organization_id);
CREATE INDEX door_event_aggregates_zone_id_88465020 ON public.door_event_aggregates USING btree (zone_id);


-- public."role" definition

-- Drop table

-- DROP TABLE "role";

CREATE TABLE "role" (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"key" varchar(250) NOT NULL,
	"name" varchar(250) NOT NULL,
	organization_id int8 NULL,
	zone_id int8 NULL,
	visible_to_customers bool NOT NULL,
	CONSTRAINT role_key_key UNIQUE (key),
	CONSTRAINT role_pkey PRIMARY KEY (id),
	CONSTRAINT role_organization_id_d00f912a_fk_api_organization_id FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT role_zone_id_dea2f53a_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX role_key_309bec13_like ON public.role USING btree (key varchar_pattern_ops);
CREATE INDEX role_organization_id_d00f912a ON public.role USING btree (organization_id);
CREATE INDEX role_zone_id_dea2f53a ON public.role USING btree (zone_id);


-- public.role_permission definition

-- Drop table

-- DROP TABLE role_permission;

CREATE TABLE role_permission (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	permission_key varchar(100) NOT NULL,
	assigned_at timestamptz NOT NULL,
	removed_at timestamptz NULL,
	assigned_by_id int4 NULL,
	removed_by_id int4 NULL,
	role_id int8 NOT NULL,
	CONSTRAINT role_permission_pkey PRIMARY KEY (id),
	CONSTRAINT role_permission_assigned_by_id_d7a6a23d_fk_auth_user_id FOREIGN KEY (assigned_by_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT role_permission_removed_by_id_dce55ff0_fk_auth_user_id FOREIGN KEY (removed_by_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT role_permission_role_id_877a80a4_fk_role_id FOREIGN KEY (role_id) REFERENCES "role"(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX role_permission_assigned_by_id_d7a6a23d ON public.role_permission USING btree (assigned_by_id);
CREATE INDEX role_permission_removed_by_id_dce55ff0 ON public.role_permission USING btree (removed_by_id);
CREATE INDEX role_permission_role_id_877a80a4 ON public.role_permission USING btree (role_id);


-- public.user_role definition

-- Drop table

-- DROP TABLE user_role;

CREATE TABLE user_role (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	assigned_at timestamptz NOT NULL,
	removed_at timestamptz NULL,
	assigned_by_id int4 NULL,
	removed_by_id int4 NULL,
	role_id int8 NOT NULL,
	user_id int4 NOT NULL,
	CONSTRAINT user_role_pkey PRIMARY KEY (id),
	CONSTRAINT user_role_assigned_by_id_35a6b317_fk_auth_user_id FOREIGN KEY (assigned_by_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT user_role_removed_by_id_2958c0d8_fk_auth_user_id FOREIGN KEY (removed_by_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT user_role_role_id_6a11361a_fk_role_id FOREIGN KEY (role_id) REFERENCES "role"(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT user_role_user_id_12d84374_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX user_role_assigned_by_id_35a6b317 ON public.user_role USING btree (assigned_by_id);
CREATE INDEX user_role_removed_by_id_2958c0d8 ON public.user_role USING btree (removed_by_id);
CREATE INDEX user_role_role_id_6a11361a ON public.user_role USING btree (role_id);
CREATE INDEX user_role_user_id_12d84374 ON public.user_role USING btree (user_id);


-- public.zone_compliance_type definition

-- Drop table

-- DROP TABLE zone_compliance_type;

CREATE TABLE zone_compliance_type (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	enabled bool NOT NULL,
	name_override varchar(250) NULL,
	compliance_type_id int8 NOT NULL,
	zone_id int8 NOT NULL,
	CONSTRAINT zone_compliance_type_compliance_type_id_zone_id_5b8ed71d_uniq UNIQUE (compliance_type_id, zone_id),
	CONSTRAINT zone_compliance_type_pkey PRIMARY KEY (id),
	CONSTRAINT zone_compliance_type_compliance_type_id_71f9174f_fk_complianc FOREIGN KEY (compliance_type_id) REFERENCES compliance_type(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT zone_compliance_type_zone_id_36e465f2_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX zone_compliance_type_compliance_type_id_71f9174f ON public.zone_compliance_type USING btree (compliance_type_id);
CREATE INDEX zone_compliance_type_zone_id_36e465f2 ON public.zone_compliance_type USING btree (zone_id);


-- public.api_incident definition

-- Drop table

-- DROP TABLE api_incident;

CREATE TABLE api_incident (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	title varchar(100) NOT NULL,
	"timestamp" timestamptz NOT NULL,
	status varchar(20) NOT NULL,
	"data" jsonb NULL,
	organization_id int8 NULL,
	priority varchar(20) NOT NULL,
	deleted_at timestamptz NULL,
	incident_type_id int8 NULL,
	invalid_feedback_count int2 NOT NULL,
	unsure_feedback_count int2 NOT NULL,
	valid_feedback_count int2 NOT NULL,
	experimental bool NOT NULL,
	uuid uuid NULL,
	review_level varchar(20) NOT NULL,
	camera_id int8 NULL,
	zone_id int8 NULL,
	corrupt_feedback_count int2 NOT NULL,
	visible_to_customers bool NOT NULL,
	alerted bool NOT NULL,
	CONSTRAINT api_incident_corrupt_feedback_count_check CHECK ((corrupt_feedback_count >= 0)),
	CONSTRAINT api_incident_invalid_feedback_count_check CHECK ((invalid_feedback_count >= 0)),
	CONSTRAINT api_incident_pkey PRIMARY KEY (id),
	CONSTRAINT api_incident_unsure_feedback_count_check CHECK ((unsure_feedback_count >= 0)),
	CONSTRAINT api_incident_uuid_key UNIQUE (uuid),
	CONSTRAINT api_incident_valid_feedback_count_check CHECK ((valid_feedback_count >= 0)),
	CONSTRAINT api_incident_camera_id_ddb78f44_fk_camera_id FOREIGN KEY (camera_id) REFERENCES camera(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_incident_incident_type_id_be74a911_fk_api_incidenttype_id FOREIGN KEY (incident_type_id) REFERENCES api_incidenttype(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_incident_organization_id_6b0ae5e6_fk_api_organization_id FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_incident_zone_id_e84b6d99_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_inciden_timesta_ba87a6_idx ON public.api_incident USING btree ("timestamp" DESC);
CREATE INDEX api_incident_camera_id_ddb78f44 ON public.api_incident USING btree (camera_id);
CREATE INDEX api_incident_new_incident_type_id_10a5d555 ON public.api_incident USING btree (incident_type_id);
CREATE INDEX api_incident_organization_id_6b0ae5e6 ON public.api_incident USING btree (organization_id);
CREATE INDEX api_incident_zone_id_e84b6d99 ON public.api_incident USING btree (zone_id);


-- public.api_incidentfeedback definition

-- Drop table

-- DROP TABLE api_incidentfeedback;

CREATE TABLE api_incidentfeedback (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	feedback_type varchar(100) NOT NULL,
	feedback_value varchar(100) NOT NULL,
	feedback_text text NULL,
	incident_id int8 NOT NULL,
	user_id int4 NOT NULL,
	deleted_at timestamptz NULL,
	CONSTRAINT api_incidentfeedback_pkey PRIMARY KEY (id),
	CONSTRAINT api_incidentfeedback_incident_id_53b0f8bd_fk FOREIGN KEY (incident_id) REFERENCES api_incident(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_incidentfeedback_user_id_3d7ffaa7_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_incidentfeedback_incident_id_53b0f8bd ON public.api_incidentfeedback USING btree (incident_id);
CREATE INDEX api_incidentfeedback_user_id_3d7ffaa7 ON public.api_incidentfeedback USING btree (user_id);


-- public.api_invitation definition

-- Drop table

-- DROP TABLE api_invitation;

CREATE TABLE api_invitation (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"token" varchar(100) NULL,
	expires_at timestamptz NULL,
	invited_by_id int4 NULL,
	invitee_id int4 NULL,
	organization_id int8 NULL,
	redeemed bool NULL,
	role_id int8 NULL,
	CONSTRAINT api_invitation_pkey PRIMARY KEY (id),
	CONSTRAINT api_invitation_invited_by_id_5c3b7c72_fk_auth_user_id FOREIGN KEY (invited_by_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_invitation_invitee_id_5a763a43_fk_auth_user_id FOREIGN KEY (invitee_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_invitation_organization_id_a1ccdc95_fk_api_organization_id FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_invitation_role_id_1f8764ad_fk_role_id FOREIGN KEY (role_id) REFERENCES "role"(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_invitation_invited_by_id_5c3b7c72 ON public.api_invitation USING btree (invited_by_id);
CREATE INDEX api_invitation_invitee_id_5a763a43 ON public.api_invitation USING btree (invitee_id);
CREATE INDEX api_invitation_organization_id_a1ccdc95 ON public.api_invitation USING btree (organization_id);
CREATE INDEX api_invitation_role_id_1f8764ad ON public.api_invitation USING btree (role_id);


-- public.api_invitation_zones definition

-- Drop table

-- DROP TABLE api_invitation_zones;

CREATE TABLE api_invitation_zones (
	id bigserial NOT NULL,
	invitation_id int8 NOT NULL,
	zone_id int8 NOT NULL,
	CONSTRAINT api_invitation_zones_invitation_id_zone_id_975324d4_uniq UNIQUE (invitation_id, zone_id),
	CONSTRAINT api_invitation_zones_pkey PRIMARY KEY (id),
	CONSTRAINT api_invitation_zones_invitation_id_d9058d84_fk_api_invit FOREIGN KEY (invitation_id) REFERENCES api_invitation(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_invitation_zones_zone_id_d34b38ff_fk_zones_id FOREIGN KEY (zone_id) REFERENCES zones(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_invitation_zones_invitation_id_d9058d84 ON public.api_invitation_zones USING btree (invitation_id);
CREATE INDEX api_invitation_zones_zone_id_d34b38ff ON public.api_invitation_zones USING btree (zone_id);


-- public.api_list_incidents definition

-- Drop table

-- DROP TABLE api_list_incidents;

CREATE TABLE api_list_incidents (
	id bigserial NOT NULL,
	list_id int8 NOT NULL,
	incident_id int8 NOT NULL,
	CONSTRAINT api_list_incidents_list_id_incident_id_e319ac17_uniq UNIQUE (list_id, incident_id),
	CONSTRAINT api_list_incidents_pkey PRIMARY KEY (id),
	CONSTRAINT api_list_incidents_incident_id_27742a4b_fk FOREIGN KEY (incident_id) REFERENCES api_incident(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_list_incidents_list_id_735e8839_fk FOREIGN KEY (list_id) REFERENCES api_list(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_list_incidents_incident_id_27742a4b ON public.api_list_incidents USING btree (incident_id);
CREATE INDEX api_list_incidents_list_id_735e8839 ON public.api_list_incidents USING btree (list_id);


-- public.api_notificationlog definition

-- Drop table

-- DROP TABLE api_notificationlog;

CREATE TABLE api_notificationlog (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"data" jsonb NULL,
	sent_at timestamptz NOT NULL,
	user_id int4 NULL,
	incident_id int8 NULL,
	category varchar(30) NULL,
	CONSTRAINT api_notificationlog_pkey PRIMARY KEY (id),
	CONSTRAINT api_notificationlog_incident_id_3766cc43_fk_api_incident_id FOREIGN KEY (incident_id) REFERENCES api_incident(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_notificationlog_user_id_e7158509_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_notificationlog_incident_id_3766cc43 ON public.api_notificationlog USING btree (incident_id);
CREATE INDEX api_notificationlog_user_id_e7158509 ON public.api_notificationlog USING btree (user_id);


-- public.api_userincident definition

-- Drop table

-- DROP TABLE api_userincident;

CREATE TABLE api_userincident (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	note varchar(1000) NULL,
	assigned_by_id int4 NULL,
	assignee_id int4 NULL,
	incident_id int8 NOT NULL,
	organization_id int8 NULL,
	CONSTRAINT api_userincident_incident_id_assignee_id_1ab74acf_uniq UNIQUE (incident_id, assignee_id),
	CONSTRAINT api_userincident_pkey PRIMARY KEY (id),
	CONSTRAINT api_userincident_assigned_by_id_11ff5513_fk_auth_user_id FOREIGN KEY (assigned_by_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_userincident_assignee_id_f9fa864c_fk_auth_user_id FOREIGN KEY (assignee_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_userincident_incident_id_4bb6e091_fk_api_incident_id FOREIGN KEY (incident_id) REFERENCES api_incident(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_userincident_organization_id_a9ba5ef7_fk_api_organ FOREIGN KEY (organization_id) REFERENCES api_organization(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_userincident_assigned_by_id_11ff5513 ON public.api_userincident USING btree (assigned_by_id);
CREATE INDEX api_userincident_assignee_id_f9fa864c ON public.api_userincident USING btree (assignee_id);
CREATE INDEX api_userincident_incident_id_4bb6e091 ON public.api_userincident USING btree (incident_id);
CREATE INDEX api_userincident_organization_id_a9ba5ef7 ON public.api_userincident USING btree (organization_id);


-- public.api_comment definition

-- Drop table

-- DROP TABLE api_comment;

CREATE TABLE api_comment (
	id bigserial NOT NULL,
	created_at timestamptz NOT NULL,
	updated_at timestamptz NOT NULL,
	deleted_at timestamptz NULL,
	"text" varchar(1000) NOT NULL,
	incident_id int8 NULL,
	owner_id int4 NULL,
	activity_type varchar(100) NULL,
	note varchar(1000) NULL,
	CONSTRAINT api_comment_pkey PRIMARY KEY (id),
	CONSTRAINT api_comment_incident_id_ea553ce5_fk FOREIGN KEY (incident_id) REFERENCES api_incident(id) DEFERRABLE INITIALLY DEFERRED,
	CONSTRAINT api_comment_owner_id_2ff90561_fk_auth_user_id FOREIGN KEY (owner_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX api_comment_incident_id_ea553ce5 ON public.api_comment USING btree (incident_id);
CREATE INDEX api_comment_owner_id_2ff90561 ON public.api_comment USING btree (owner_id);


-- public.auth_group_permissions definition

-- Drop table

-- DROP TABLE auth_group_permissions;

CREATE TABLE auth_group_permissions (
	id bigserial NOT NULL,
	group_id int4 NOT NULL,
	permission_id int4 NOT NULL,
	CONSTRAINT auth_group_permissions_group_id_permission_id_0cd325b0_uniq UNIQUE (group_id, permission_id),
	CONSTRAINT auth_group_permissions_pkey PRIMARY KEY (id)
);
CREATE INDEX auth_group_permissions_group_id_b120cbf9 ON public.auth_group_permissions USING btree (group_id);
CREATE INDEX auth_group_permissions_permission_id_84c5c92e ON public.auth_group_permissions USING btree (permission_id);


-- public.auth_permission definition

-- Drop table

-- DROP TABLE auth_permission;

CREATE TABLE auth_permission (
	id serial4 NOT NULL,
	"name" varchar(255) NOT NULL,
	content_type_id int4 NOT NULL,
	codename varchar(100) NOT NULL,
	CONSTRAINT auth_permission_content_type_id_codename_01ab375a_uniq UNIQUE (content_type_id, codename),
	CONSTRAINT auth_permission_pkey PRIMARY KEY (id)
);
CREATE INDEX auth_permission_content_type_id_2f476e4b ON public.auth_permission USING btree (content_type_id);


-- public.auth_user_user_permissions definition

-- Drop table

-- DROP TABLE auth_user_user_permissions;

CREATE TABLE auth_user_user_permissions (
	id bigserial NOT NULL,
	user_id int4 NOT NULL,
	permission_id int4 NOT NULL,
	CONSTRAINT auth_user_user_permissions_pkey PRIMARY KEY (id),
	CONSTRAINT auth_user_user_permissions_user_id_permission_id_14a6b632_uniq UNIQUE (user_id, permission_id)
);
CREATE INDEX auth_user_user_permissions_permission_id_1fbb5f2c ON public.auth_user_user_permissions USING btree (permission_id);
CREATE INDEX auth_user_user_permissions_user_id_a95ead1b ON public.auth_user_user_permissions USING btree (user_id);


-- public.auth_group_permissions foreign keys

ALTER TABLE public.auth_group_permissions ADD CONSTRAINT auth_group_permissio_permission_id_84c5c92e_fk_auth_perm FOREIGN KEY (permission_id) REFERENCES auth_permission(id) DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE public.auth_group_permissions ADD CONSTRAINT auth_group_permissions_group_id_b120cbf9_fk_auth_group_id FOREIGN KEY (group_id) REFERENCES auth_group(id) DEFERRABLE INITIALLY DEFERRED;


-- public.auth_permission foreign keys

ALTER TABLE public.auth_permission ADD CONSTRAINT auth_permission_content_type_id_2f476e4b_fk_django_co FOREIGN KEY (content_type_id) REFERENCES django_content_type(id) DEFERRABLE INITIALLY DEFERRED;


-- public.auth_user_user_permissions foreign keys

ALTER TABLE public.auth_user_user_permissions ADD CONSTRAINT auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm FOREIGN KEY (permission_id) REFERENCES auth_permission(id) DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE public.auth_user_user_permissions ADD CONSTRAINT auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES auth_user(id) DEFERRABLE INITIALLY DEFERRED;