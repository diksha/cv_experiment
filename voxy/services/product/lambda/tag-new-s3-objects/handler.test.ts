import { handler } from "./handler";
import { RETENTION_POLICY_TAG_KEY, RETENTION_POLICY_TAG_VALUE_STANDARD } from "./constants";
import { S3Client } from "@aws-sdk/client-s3";
import { S3EventRecord } from "aws-lambda";
import { mockClient } from "aws-sdk-client-mock";

const s3ClientMock = mockClient(S3Client);

describe("tag-new-s3-objects", () => {
  beforeEach(() => {
    s3ClientMock.reset();
  });

  it("returns 400 response for empty records list", async () => {
    // Arrange
    const event = { Records: [] };

    // Act
    const result = await handler(event);

    // Assert
    expect(result.statusCode).toBe(400);
  });

  it("tags single object", async () => {
    // Arrange
    const event = {
      Records: [
        {
          s3: {
            bucket: { name: "bucket_1" },
            object: { key: "key_1" },
          },
        } as S3EventRecord,
      ],
    };

    // Act
    const result = await handler(event);

    // Assert
    const expectedRequestArgs = {
      Bucket: "bucket_1",
      Key: "key_1",
      Tagging: {
        TagSet: [
          {
            Key: RETENTION_POLICY_TAG_KEY,
            Value: RETENTION_POLICY_TAG_VALUE_STANDARD,
          },
        ],
      },
    };
    expect(result.statusCode).toBe(200);
    expect(s3ClientMock.call(0).args[0].input).toEqual(expectedRequestArgs);
  });

  it("tags multiple objects", async () => {
    // Arrange
    const bucketKeyPairs = [
      { bucket: "bucket_1", key: "key_1" },
      { bucket: "bucket_2", key: "key_2" },
      { bucket: "bucket_3", key: "key_3" },
    ];

    const event = {
      Records: bucketKeyPairs.map(
        (bucketKeyPair) =>
          ({
            s3: {
              bucket: { name: bucketKeyPair.bucket },
              object: { key: bucketKeyPair.key },
            },
          } as S3EventRecord)
      ),
    };

    // Act
    const result = await handler(event);

    // Assert
    const expectedRequestArgs = bucketKeyPairs.map((bucketKeyPair) => ({
      Bucket: bucketKeyPair.bucket,
      Key: bucketKeyPair.key,
      Tagging: {
        TagSet: [
          {
            Key: RETENTION_POLICY_TAG_KEY,
            Value: RETENTION_POLICY_TAG_VALUE_STANDARD,
          },
        ],
      },
    }));

    expect(result.statusCode).toBe(200);

    for (let i = 0; i < bucketKeyPairs.length; i++) {
      expect(s3ClientMock.call(i).args[0].input).toEqual(expectedRequestArgs[i]);
    }
  });
});
