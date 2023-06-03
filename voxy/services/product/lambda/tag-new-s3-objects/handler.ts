import {
  S3Client,
  PutObjectTaggingRequest,
  PutObjectTaggingCommand,
  PutObjectTaggingCommandOutput,
} from "@aws-sdk/client-s3";
import { S3Event } from "aws-lambda";
import { Response } from "./types";
import { RETENTION_POLICY_TAG_KEY, RETENTION_POLICY_TAG_VALUE_STANDARD } from "./constants";

const s3Client = new S3Client({});

/**
 * Handle invocations of the function via S3 events.
 *
 * @remarks
 * AWS example code only demonstrates handling a single record, but the S3Event
 * type definition supports multiple records, so we'll handle that case by
 * iterating records, sending a tag request for each record, and awaiting all
 * promise resolutions.
 *
 * @param event - S3 event
 * @returns invocation response
 */
export const handler = async (event: S3Event): Promise<Response> => {
  // export const handler = async (event: S3EventInterface): Promise<Response> => {
  const { Records: records } = event;
  if (records?.length === 0) {
    return buildResponse(400, "No records found in event");
  }

  const taggedObjects: string[] = [];
  const tagPromises = records.map((record) => {
    const bucket = record.s3.bucket.name;
    const key = sanitizeS3EventObjectKey(record.s3.object.key);
    taggedObjects.push(`${bucket}/${key}`);
    return tagObject(key, bucket);
  });

  await Promise.all(tagPromises);
  const responseMessage = `Tagging complete for object(s): ${taggedObjects.join(", ")}`;
  return buildResponse(200, responseMessage);
};

/**
 * Tag an S3 object with default tags.
 *
 * @param key - S3 object key
 * @param bucket  - s3 object bucket name
 * @returns tag command promise
 */
function tagObject(key: string, bucket: string): Promise<PutObjectTaggingCommandOutput> {
  const request: PutObjectTaggingRequest = {
    Bucket: bucket,
    Key: key,
    Tagging: {
      TagSet: [
        {
          Key: RETENTION_POLICY_TAG_KEY,
          Value: RETENTION_POLICY_TAG_VALUE_STANDARD,
        },
      ],
    },
  };

  const command = new PutObjectTaggingCommand(request);
  return s3Client.send(command);
}

/**
 * Build a handler response object.
 *
 * @param statusCode - response status code
 * @param message - response message
 * @returns response object
 */
function buildResponse(statusCode: number, message: string): Response {
  return {
    statusCode,
    body: JSON.stringify({
      message,
    }),
  };
}

/**
 * Sanitize S3 object key from S3 event.
 *
 * @remarks
 * S3 event object keys are URI encoded, so we need to decode them.
 *
 * @param key - S3 object key from S3 event
 * @returns sanitized key
 */
function sanitizeS3EventObjectKey(key: string): string {
  return decodeURIComponent(key.replace(/\+/g, " "));
}
