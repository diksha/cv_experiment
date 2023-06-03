--
-- Copyright 2020-2021 Voxel Labs, Inc.
-- All rights reserved.
--
-- This document may not be reproduced, republished, distributed, transmitted,
-- displayed, broadcast or otherwise exploited in any manner without the express
-- prior written permission of Voxel Labs, Inc. The receipt or possession of this
-- document does not convey any rights to reproduce, disclose, or distribute its
-- contents, or to manufacture, use, or sell anything that it may describe, in
-- whole or in part.
--

-- Syncs incident feedback count columns with actual feedback data
WITH counts AS (
    SELECT inner_incident.id       AS incident_id,
        COUNT(valid_feedback.id)   AS valid_feedback_count,
        COUNT(invalid_feedback.id) AS invalid_feedback_count,
        COUNT(unsure_feedback.id)  AS unsure_feedback_count
    FROM api_incident inner_incident
        LEFT JOIN api_incidentfeedback valid_feedback
            ON inner_incident.id = valid_feedback.incident_id
            AND valid_feedback.feedback_value = 'valid'
        LEFT JOIN api_incidentfeedback invalid_feedback
            ON inner_incident.id = invalid_feedback.incident_id
            AND invalid_feedback.feedback_value = 'invalid'
        LEFT JOIN api_incidentfeedback unsure_feedback
            ON inner_incident.id = unsure_feedback.incident_id
            AND unsure_feedback.feedback_value = 'unsure'
    GROUP BY inner_incident.id
)
UPDATE api_incident
SET
    valid_feedback_count = counts.valid_feedback_count,
    invalid_feedback_count = counts.invalid_feedback_count,
    unsure_feedback_count = counts.unsure_feedback_count
FROM counts
WHERE api_incident.id = counts.incident_id
;

-- Ad-hoc queries to sanity test the results
SELECT COUNT(*) FROM api_incident WHERE valid_feedback_count > 0;
SELECT COUNT(*) FROM api_incident WHERE invalid_feedback_count > 0;
SELECT COUNT(*) FROM api_incident WHERE unsure_feedback_count > 0;
