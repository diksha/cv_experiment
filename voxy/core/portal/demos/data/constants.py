from core.portal.scores.graphql.types import Score

DEMO_ORG_KEY = "VOXEL_DEMO"
DEMO_SITE_ATLANTA_NAME = "Atlanta"
DEMO_SITE_MARKET_NAME = "Market St."
DEMO_SITE_HARRISON_NAME = "Harrison St."
DEMO_SITE_KEY = "DEMO"
DEMO_CAMERA_UUID_PREFIX = f"{DEMO_ORG_KEY}/{DEMO_SITE_KEY}".lower()
DEMO_SCORE_DATA = {
    "organizations": {
        DEMO_ORG_KEY: {
            "overallScore": Score(value=73, label="Voxel Demo"),
            "eventScores": [
                Score(value=58, label="No Stop At Intersection"),
                Score(value=79, label="Improper Bend"),
                Score(value=86, label="Piggyback"),
                Score(value=83, label="Hard Hat"),
                Score(value=87, label="Safety Vest"),
                Score(value=78, label="Parking Duration"),
                Score(value=29, label="No Ped Zone"),
                Score(value=78, label="Overreaching"),
            ],
        },
    },
    "sites": {
        DEMO_SITE_ATLANTA_NAME: {
            "overallScore": Score(value=77, label=DEMO_SITE_ATLANTA_NAME),
            "eventScores": [
                Score(value=65, label="No Stop At Intersection"),
                Score(value=78, label="Piggyback"),
                Score(value=90, label="Hard Hat"),
                Score(value=96, label="Improper Bend"),
                Score(value=88, label="Safety Vest"),
                Score(value=93, label="Parking Duration"),
                Score(value=21, label="No Ped Zone"),
                Score(value=86, label="Overreaching"),
            ],
        },
        DEMO_SITE_HARRISON_NAME: {
            "overallScore": Score(value=71, label=DEMO_SITE_HARRISON_NAME),
            "eventScores": [
                Score(value=14, label="No Stop At Intersection"),
                Score(value=95, label="Piggyback"),
                Score(value=84, label="Hard Hat"),
                Score(value=86, label="Improper Bend"),
                Score(value=93, label="Safety Vest"),
                Score(value=78, label="Parking Duration"),
                Score(value=48, label="No Ped Zone"),
                Score(value=72, label="Overreaching"),
            ],
        },
        DEMO_SITE_MARKET_NAME: {
            "overallScore": Score(value=68, label=DEMO_SITE_MARKET_NAME),
            "eventScores": [
                Score(value=94, label="No Stop At Intersection"),
                Score(value=85, label="Piggyback"),
                Score(value=74, label="Hard Hat"),
                Score(value=56, label="Improper Bend"),
                Score(value=81, label="Safety Vest"),
                Score(value=62, label="Parking Duration"),
                Score(value=18, label="No Ped Zone"),
                Score(value=77, label="Overreaching"),
            ],
        },
    },
}
