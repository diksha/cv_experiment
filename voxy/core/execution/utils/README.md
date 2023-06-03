# Note about Polygon naming scheme of override files

Polygon override files need to be stored in S3 with a naming scheme that leads to accessiblity to people viewing the overrides and the upcoming API.
I'm proposing the scheme should be in the format of `polygon/{file_to_override}`, where `file_to_override` is the path as it appears in the repo.

This will be useful since:

- The location of the `cha.yaml` files is not subject to frequent change
- The format piggybacks off of what the team is already familiar with
- This format guarantees uniqueness and trivializes finding the correct file as long as the file location is known

A helper func `generate_polygon_override_key` exists to contain the logic for putting this key together. This same func is used in the logic to push to S3 and in the test.

Example: `polygon/configs/cameras/hensley/chandler/0001/cha.yaml`
