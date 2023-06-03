import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { LinkBreak } from "phosphor-react";
import { BackgroundSpinner, Card, DateTimeString, StickyHeader, LogoIcon } from "ui";
import { Button } from "@mui/material";
import { Player } from "features/video";
import { BrightYellow } from "features/incidents";
import { isObject } from "lodash";

interface IncidentData {
  title: string;
  siteName: string;
  videoUrl: string;
  annotationsUrl: string;
  cameraName: string;
  timestamp: string;
  actorIds: string[];
}

const sanitizeAPIResponse = (data: unknown): IncidentData | undefined => {
  if (!isObject(data)) {
    return undefined;
  }

  const title = "title" in data && String(data["title"]);
  const timestamp = "timestamp" in data && String(data["timestamp"]);
  const siteName = "zone_name" in data && String(data["zone_name"]);
  const cameraName = "camera_name" in data && String(data["camera_name"]);
  const videoUrl = "video_url" in data && String(data["video_url"]);
  const annotationsUrl = "annotations_url" in data && String(data["annotations_url"]);
  const actorIds = "actor_ids" in data && (data["actor_ids"] as string[]);

  const valid = title && timestamp && siteName && cameraName && videoUrl && annotationsUrl && actorIds;

  if (!valid) {
    return undefined;
  }

  return {
    title,
    timestamp,
    siteName,
    cameraName,
    videoUrl,
    annotationsUrl,
    actorIds,
  };
};

export function SharedIncidentsPage() {
  const { token }: any = useParams();
  const [loading, setLoading] = useState(true);
  const [incidentData, setIncidentData] = useState<IncidentData>();
  const [hasError, setHasError] = useState<boolean>(false);

  useEffect(() => {
    const fetchIncidentData = async () => {
      try {
        const response = await fetch(`/api/share/${token}/`);
        const data = await response.json();
        const sanitizedData = sanitizeAPIResponse(data);
        setIncidentData(sanitizedData);

        if (!data) {
          setHasError(true);
        }
      } catch (err) {
        setHasError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchIncidentData();
  }, [token, setIncidentData]);

  if (loading) {
    return <BackgroundSpinner />;
  }

  return (
    <>
      {incidentData ? (
        <StickyHeader zIndex={40} sentinelClassName="h-0" flush={true}>
          <div className="bg-white py-4 border-b border-brand-gray-050">
            <div className="px-0 md:px-8">
              <div className="flex gap-4 px-4 md:px-0 justify-between content-center items-center">
                <div className="pt-1 font-epilogue font-bold text-lg text-brand-gray-500 flex items-center justify-center">
                  <div className="mx-4 w-8 h-8 flex justify-center items-center">
                    <LogoIcon />
                  </div>
                  <div className="font-epilogue font-bold text-brand-gray-500 md:text-lg">{incidentData.siteName}</div>
                </div>
                <div className="hidden md:block">
                  <Button variant="outlined" href="/">
                    Log in
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </StickyHeader>
      ) : null}
      {hasError ? (
        <div className="px-4 flex items-center justify-center h-screen w-full">
          <div className="m-auto flex flex-col items-center justify-center p-5 py-10 w-96">
            <div>
              <LinkBreak size={64} color="##676785" weight="fill" />
            </div>
            <p className="font-bold text-3xl font-epilogue m-2">Link Expired</p>
            <p className="text-gray-600 text-center m-2">
              Event links expire after 3 days. Please ask the sender for a new link.
            </p>
          </div>
        </div>
      ) : null}
      <div className="p-0 md:px-8 md:py-8">
        <Card noPadding className="m-auto w-full md:max-w-4xl">
          {incidentData ? (
            <div className="flex flex-col h-full">
              <div className="flex flex-col items-center">
                <div className="flex-grow m-2 py-2 px-4 w-full">
                  <Player
                    videoUrl={incidentData.videoUrl}
                    annotationsUrl={incidentData.annotationsUrl}
                    actorIds={incidentData.actorIds}
                    annotationColorHex={BrightYellow}
                    controls
                  />
                  <div className="mt-4">
                    <div className="flex justify-between">
                      <div className="font-epilogue font-bold text-lg">{incidentData.title}</div>
                      <div className="text-md">
                        <DateTimeString dateTime={incidentData.timestamp} includeTimezone />
                      </div>
                    </div>
                    <div className="text-md">{incidentData.cameraName}</div>
                  </div>
                </div>
              </div>
              <div className="w-full flex flex-col items-center justify-center z-10 md:-mb-12">
                <div className="font-bold mb-4">Already using Voxel?</div>
                <Button variant="contained" href="/" size="small">
                  Log in
                </Button>
              </div>
              <BlurredSkeleton />
            </div>
          ) : null}
        </Card>
      </div>
    </>
  );
}

function BlurredSkeleton() {
  return (
    <div className="flex flex-col gap-14 px-6 pb-16 blur-md z-0">
      <div className="flex gap-8">
        <div className="flex-1"></div>
        <div className="h-12 w-12 bg-brand-green-200 rounded-full"></div>
        <div className="h-12 w-12 bg-brand-red-200 rounded-full"></div>
      </div>
      <div className="flex gap-4">
        <div className="h-12 w-12 bg-brand-gray-100"></div>
        <div className="h-12 w-12 bg-brand-gray-100"></div>
        <div className="h-12 w-12 bg-brand-gray-100"></div>
        <div className="h-12 w-12 bg-brand-gray-100"></div>
      </div>
      <div className="flex gap-4">
        <div className="h-16 w-16 bg-brand-gray-200 rounded-full"></div>
        <div className="flex flex-col gap-4 flex-grow">
          <div className="bg-gray-700 h-2 w-1/3"></div>
          <div className="bg-gray-400 h-2 w-full"></div>
          <div className="bg-gray-400 h-2 w-2/3"></div>
        </div>
      </div>
      <div className="flex gap-4">
        <div className="h-16 w-16 bg-brand-gray-100 rounded-full"></div>
        <div className="flex flex-col gap-4 flex-grow">
          <div className="bg-gray-700 h-2 w-1/3"></div>
          <div className="bg-gray-400 h-2 w-2/3"></div>
        </div>
      </div>
      <div className="flex gap-4">
        <div className="h-16 w-16 bg-brand-purple-100 rounded-full"></div>
        <div className="flex flex-col gap-4 flex-grow">
          <div className="bg-gray-700 h-2 w-1/3"></div>
          <div className="bg-gray-400 h-2 w-full"></div>
          <div className="bg-gray-400 h-2 w-full"></div>
          <div className="bg-gray-400 h-2 w-1/2"></div>
        </div>
      </div>
      <div className="flex gap-4">
        <div className="w-16"></div>
        <div className="flex-1 h-16 border-3 border-brand-gray-300"> </div>
      </div>
      <div className="flex">
        <div className="flex-1"></div>
        <div className="bg-brand-gray-200 w-32 h-12"></div>
      </div>
    </div>
  );
}
