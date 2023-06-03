-- Set api_incident.last_feedback_submission_timestamp to the timestamp
-- of the most recent incident feedback submission timestamp

WITH latest_feedback_timestamps AS (
    SELECT
        incident_id,
        max(created_at) AS "latest_timestamp"
    FROM api_incidentfeedback
    GROUP BY incident_id
)
UPDATE api_incident
SET last_feedback_submission_timestamp = feedback.latest_timestamp
FROM latest_feedback_timestamps feedback
WHERE api_incident.id = feedback.incident_id
;
