------------------------------------------
-- Cleanup test data
------------------------------------------
delete from state_state where camera_uuid = '66665dd9-91ff-482c-a108-6bc2330872d9';

------------------------------------------
-- Uptime data
------------------------------------------
INSERT INTO state_state (timestamp, end_timestamp, organization, location, zone, camera_uuid, camera_name, actor_category, actor_id, motion_zone_is_in_motion) VALUES ('2022-11-16 1:00:00', '2022-11-16 1:30:00', 'acme-org', 'acme-site', 'acme-zone', '66665dd9-91ff-482c-a108-6bc2330872d9', 'Camera 1', 19, 'actor-1', TRUE);
INSERT INTO state_state (timestamp, end_timestamp, organization, location, zone, camera_uuid, camera_name, actor_category, actor_id, motion_zone_is_in_motion) VALUES ('2022-11-16 2:00:00', '2022-11-16 3:30:00', 'acme-org', 'acme-site', 'acme-zone', '66665dd9-91ff-482c-a108-6bc2330872d9', 'Camera 1', 19, 'actor-1', TRUE);
INSERT INTO state_state (timestamp, end_timestamp, organization, location, zone, camera_uuid, camera_name, actor_category, actor_id, motion_zone_is_in_motion) VALUES ('2022-11-16 5:30:00', '2022-11-16 5:45:00', 'acme-org', 'acme-site', 'acme-zone', '66665dd9-91ff-482c-a108-6bc2330872d9', 'Camera 1', 19, 'actor-1', TRUE);

------------------------------------------
-- Downtime data
------------------------------------------
INSERT INTO state_state (timestamp, end_timestamp, organization, location, zone, camera_uuid, camera_name, actor_category, actor_id, motion_zone_is_in_motion) VALUES ('2022-11-16 1:30:00', '2022-11-16 2:00:00', 'acme-org', 'acme-site', 'acme-zone', '66665dd9-91ff-482c-a108-6bc2330872d9', 'Camera 1', 19, 'actor-1', FALSE);
INSERT INTO state_state (timestamp, end_timestamp, organization, location, zone, camera_uuid, camera_name, actor_category, actor_id, motion_zone_is_in_motion) VALUES ('2022-11-16 3:30:00', '2022-11-16 5:30:00', 'acme-org', 'acme-site', 'acme-zone', '66665dd9-91ff-482c-a108-6bc2330872d9', 'Camera 1', 19, 'actor-1', FALSE);
INSERT INTO state_state (timestamp, end_timestamp, organization, location, zone, camera_uuid, camera_name, actor_category, actor_id, motion_zone_is_in_motion) VALUES ('2022-11-16 5:45:00', '2022-11-16 6:00:00', 'acme-org', 'acme-site', 'acme-zone', '66665dd9-91ff-482c-a108-6bc2330872d9', 'Camera 1', 19, 'actor-1', FALSE);

------------------------------------------
-- Expected results
------------------------------------------
-- hour                                 actor_id    uptime_duration_s   downtime_duration_s     max_timestamp
-- 2022-11-16 01:00:00.000000 +00:00    actor-1     1800                1800                    2022-11-16 01:30:00.000000 +00:00
-- 2022-11-16 02:00:00.000000 +00:00    actor-1     3600                0                       2022-11-16 02:00:00.000000 +00:00
-- 2022-11-16 03:00:00.000000 +00:00    actor-1     1800                1800                    2022-11-16 03:30:00.000000 +00:00
-- 2022-11-16 04:00:00.000000 +00:00    actor-1     0                   3600                    2022-11-16 03:30:00.000000 +00:00
-- 2022-11-16 05:00:00.000000 +00:00    actor-1     900                 2700                    2022-11-16 05:45:00.000000 +00:00


WITH time_buckets AS (
    SELECT generate_series(
        date_trunc('hour', '2022-11-15 00:00:00'::timestamp),
        date_trunc('hour', '2022-11-18 00:00:00'::timestamp),
        '1 hour'::INTERVAL
    ) AS hour
)
-- All production line actors
, actors AS (
    SELECT DISTINCT actor_id
    FROM state_state
    WHERE
        actor_category = 19
        AND actor_id <> ''
        AND timestamp BETWEEN '2022-11-15 00:00:00'::timestamp AND '2022-11-18 00:00:00'::timestamp
)
-- All actor * time bucket combinations
, actor_time_buckets AS (
    SELECT *
    FROM time_buckets
    CROSS JOIN actors
)
SELECT
    atb.hour,
    atb.actor_id,
    SUM(
        CASE WHEN s.motion_zone_is_in_motion IS TRUE THEN
            EXTRACT(epoch FROM (
                LEAST(atb.hour + '1 hour', s.timestamp + (s.end_timestamp - s.timestamp)) - GREATEST(atb.hour, s.timestamp)
            ))
        ELSE 0 END
    ) AS "uptime_duration_s",
    SUM(
        CASE WHEN s.motion_zone_is_in_motion IS FALSE THEN
            EXTRACT(epoch FROM (
                LEAST(atb.hour + '1 hour', s.timestamp + (s.end_timestamp - s.timestamp)) - GREATEST(atb.hour, s.timestamp)
            ))
        ELSE 0 END
    ) AS "downtime_duration_s",
    MAX(s.timestamp) AS "max_timestamp"
FROM
    actor_time_buckets atb
    LEFT JOIN state_state s
        ON atb.actor_id = s.actor_id
        AND s.actor_category = 19
        AND s.timestamp < atb.hour + '1 hour'
        AND s.end_timestamp >= atb.hour
WHERE
    -- Only include records where the state time range overlaps with our target start/end range
    s.timestamp < '2022-11-18 00:00:00'::timestamp
    OR s.end_timestamp >= '2022-11-15 00:00:00'::timestamp
GROUP BY
    atb.hour,
    atb.actor_id
;

