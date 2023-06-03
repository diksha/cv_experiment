/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";

import { AVLTree } from "shared/dataStructures/AVLTree";
import { generateColorStack } from "shared/utilities/colors";
import styles from "./Player.module.css";
import "video.js/dist/video-js.min.css";
import videojs from "video.js";
import { Box } from "@mui/material";
import { Spinner } from "ui";

const DEFAULT_ANNOTATION_COLOR_HEX = "#E74C3C";

type Point = {
  x: number;
  y: number;
  z: number;
};

type Polygon = {
  vertices: Point[];
};

type Actor = {
  track_id: string;
  track_uuid: string;
  polygon?: Polygon;
};

type Frame = {
  relative_timestamp_ms: number;
  actors: Actor[];
  frame_width: number;
  frame_height: number;
};

type Vertex = [number, number];
type PolygonConfig = {
  polygon: Vertex[];
};

enum AnnotationMode {
  All,
  None,
  Highlighted,
}

/**
 * Class used to store and look up frame data.
 *
 * This special data structure is required because we are currently polling
 * the browser video APIs for the current timestamp, which often does not map
 * directly to an annotation's timestamp. This data structure allows us to
 * efficiently look up the nearest annotation data for arbitrary timestamps.
 */
class FrameMap {
  timestampTree: AVLTree;
  frameMap: { [key: number]: Frame };

  constructor() {
    this.timestampTree = new AVLTree();
    this.frameMap = {} as { number: Frame };
  }

  insertFrame(timestamp: number, frame: Frame) {
    this.timestampTree.insert(timestamp);
    this.frameMap[timestamp] = frame;
  }

  /**
   * Gets frame annotation data for the provided timestamp.
   */
  getFrame(timestamp: number): Frame | undefined {
    // Traverse timestamp tree to find nearest matching timestamp
    const nearestTimestamp = this.timestampTree.findNearest(timestamp, "ceiling");

    // Return the matching frame data (if any)
    return this.frameMap?.[nearestTimestamp || -1];
  }
}

const usePlayer = (props: { videoUrl: string | string[]; controls: boolean }) => {
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);

  const options = {
    fill: true,
    fluid: true,
    playsinline: true,
    autoplay: false,
    preferFullWindow: true,
    bigPlayButton: false,
    html5: {
      hls: {
        enableLowInitialPlaylist: true,
        smoothQualityChange: true,
        overrideNative: true,
      },
    },
  };
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [player, setPlayer] = useState<videojs.Player | undefined>();

  useEffect(() => {
    const vjsPlayer = videojs(videoRef.current!, {
      ...options,
      controls: props.controls,
      playbackRates: [0.5, 1, 1.5, 2, 3, 4, 5],
      controlBar: {
        pictureInPictureToggle: false,
      },
    });

    vjsPlayer.on("ready", () => {
      setPlayer(vjsPlayer);
    });

    return () => {
      if (player) {
        player.dispose();
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (player) {
      const src = Array.isArray(props.videoUrl) ? props.videoUrl[currentVideoIndex] : props.videoUrl;

      player.src(src);
      player.aspectRatio("16:9");
      player.playsinline(true);
      // Use state so that when resizing happen, react will re-render,
      // and we can sync with canvas size
      // this only happen when video start playing
      player.on("resize", () => {
        setVideoSize({
          width: player.videoWidth(),
          height: player.videoHeight(),
        });
      });

      player.on("error", () => {
        if (Array.isArray(props.videoUrl) && currentVideoIndex < props.videoUrl.length - 1) {
          setCurrentVideoIndex(currentVideoIndex + 1);
        }
      });
    }
  }, [player, props.videoUrl, currentVideoIndex]);

  return { videoRef, videoSize };
};

export const Player = (props: {
  videoUrl: string | string[];
  annotationsUrl?: string | string[];
  controls: boolean;
  actorIds?: string[];
  // TODO: define type for config
  cameraConfig?: any;
  annotationColorHex?: string | null;
  onVideoCanPlay?: () => void;
  hideCanvas?: boolean;
  autoplay?: boolean;
}) => {
  const [playerReady, setPlayerReady] = useState(false);
  const [frameMap, setFrameMap] = useState(new FrameMap());
  const [actorMap, setActorMap] = useState({});
  const [frameAnnotations, setFrameAnnotations] = useState([]);
  const [annotationsReady, setAnnotationsReady] = useState(false);
  const [currentAnnotationsIndex, setCurrentAnnotationsIndex] = useState(0);

  const { videoRef, videoSize } = usePlayer({
    videoUrl: props.videoUrl,
    controls: props.controls,
  });
  const [annotationMode, setAnnotationMode] = useState(AnnotationMode.None);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Used for consistent rounding logic.
  // Seems to work OK for now, should be optimized for higher fidelity.
  const roundTimestamp = (timestamp: number): number => {
    return Math.round(timestamp);
  };

  const readyToPlay = videoRef?.current && playerReady && annotationsReady;
  // set autoplay to default
  const autoplay = props.autoplay !== undefined ? props.autoplay : true;

  useEffect(() => {
    const downloadAnnotations = async () => {
      if (props.annotationsUrl) {
        try {
          const url = Array.isArray(props.annotationsUrl)
            ? props.annotationsUrl[currentAnnotationsIndex]
            : props.annotationsUrl;
          const response = await fetch(url);
          const data = await response.json();

          setFrameAnnotations(data.frames);
        } catch (e) {
          if (Array.isArray(props.annotationsUrl) && currentAnnotationsIndex < props.annotationsUrl.length - 1) {
            setCurrentAnnotationsIndex(currentAnnotationsIndex + 1);
          } else {
            console.error(e);
          }
        } finally {
          setAnnotationsReady(true);
        }
      } else if (props.videoUrl) {
        // If we do NOT have an annotations URL but we do have a video URL,
        // assume no annotations URL is available and mark annotations as ready.
        setAnnotationsReady(true);
      }
    };

    downloadAnnotations();

    if (props.actorIds && props.actorIds.length > 0) {
      setAnnotationMode(AnnotationMode.Highlighted);
    }
  }, [props.actorIds, props.annotationsUrl, props.videoUrl, currentAnnotationsIndex]);

  useEffect(() => {
    let isMounted = true;
    const frameMap = new FrameMap();
    const actorMap = {} as any;
    const colors = generateColorStack();
    let timestampOffset = -1;

    if (frameAnnotations) {
      frameAnnotations.forEach((frame: Frame) => {
        // HACK: initialize offsets based on initial timestamp
        // Ideally this is handled in the controller/publisher
        if (timestampOffset === -1) {
          timestampOffset = frame.relative_timestamp_ms;
        }

        const relativeTimestampMs = roundTimestamp(frame.relative_timestamp_ms - timestampOffset);

        switch (annotationMode) {
          case AnnotationMode.Highlighted:
            // Only include highlighted actors in frame map
            frame.actors = frame?.actors.filter((actor: Actor) => {
              const matchingTrackId = props.actorIds?.includes(String(actor.track_id));
              const matchingTrackUuid = props.actorIds?.includes(String(actor.track_uuid));
              return matchingTrackId || matchingTrackUuid;
            });
            frameMap.insertFrame(relativeTimestampMs, frame);
            break;
          case AnnotationMode.All:
            // Include all actors
            frameMap.insertFrame(relativeTimestampMs, frame);
            // Generate an actor map with colors
            frame.actors.forEach((actor: any) => {
              if (!actorMap[actor.track_id]) {
                actorMap[actor.track_id] = {
                  color: colors.shift(),
                };
              }
            });
            break;
        }
      });
    }

    if (isMounted) {
      setActorMap(actorMap);
      setFrameMap(frameMap);
      setPlayerReady(true);
    }
  }, [props.actorIds, annotationMode, annotationsReady, frameAnnotations]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      const overlayCanvas = () => {
        const canvas = canvasRef?.current;
        const video = videoRef?.current;

        if (canvas && video) {
          // We need to make sure the canvas aspect ratio matches that
          // of the source video. In fullscreen, the video element
          // aspect ratio changes because the video gets letterboxed/pillarboxed
          // depending on the height/width of the viewport.
          // PRO TIP: to debug this logic, set the canvas background to transparent red.
          const parentRect = video.getBoundingClientRect();
          const parentAspectRatio = parentRect.width / parentRect.height;
          const sourceAspectRatio = video.videoWidth / video.videoHeight;

          let targetWidth = 0;
          let targetHeight = 0;

          if (props.hideCanvas && props.hideCanvas === true) {
            targetWidth = 0;
            targetHeight = 0;
          } else if (parentAspectRatio > sourceAspectRatio) {
            // Parent is "wider" than source (pillarboxing occurs)
            targetHeight = parentRect.height;
            targetWidth = parentRect.height * sourceAspectRatio;
          } else {
            // Parent is "narrower" than source (letterboxing occurs)
            targetHeight = parentRect.width / sourceAspectRatio;
            targetWidth = parentRect.width;
          }

          // These dimensions should match the source video as rendered.
          // We let CSS center it (which should align with letterbox/pillarbox).
          canvas.style.width = `${targetWidth}px`;
          canvas.style.height = `${targetHeight}px`;
        }
      };
      // Attach resize observer to keep canvas and video position in sync
      new ResizeObserver(overlayCanvas).observe(video);
    }
  }, [videoSize, videoRef, props.hideCanvas]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) {
      return;
    }

    var ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const lineWidth = 3;
    var frame = frameMap?.getFrame(frameMap?.timestampTree?.root?.value);
    var widthRatio = video.videoWidth / (frame?.frame_width ?? video.videoWidth);
    var heightRatio = video.videoHeight / (frame?.frame_height ?? video.videoHeight);

    const drawStaticPolygons = (polygons: PolygonConfig[], fillColor: string) => {
      polygons.forEach((area: PolygonConfig) => {
        ctx.fillStyle = fillColor;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(area.polygon[0][0] * video.videoWidth, area.polygon[0][1] * video.videoHeight);
        // hacky draw, assumes already sorted
        area.polygon.forEach((vertex: Vertex) => {
          ctx.lineTo(vertex[0] * video.videoWidth, vertex[1] * video.videoHeight);
        });
        ctx.closePath();
        ctx.fill();
      });
    };

    const polygonTypes = [
      "drivingAreas",
      "actionableRegions",
      "doors",
      "intersections",
      "endOfAisles",
      "noPedestrianZones",
      "motionDetectionZones",
      "noObstructionRegions",
    ];
    polygonTypes.forEach((polygonType) => {
      if (props.cameraConfig && props.cameraConfig[polygonType]) {
        drawStaticPolygons(JSON.parse(props.cameraConfig[polygonType]), "rgba(255,0,0,.1)");
      }
    });

    const drawBoundingBox = (actor: Actor) => {
      if (!actor?.polygon?.vertices) {
        return;
      }
      let color = DEFAULT_ANNOTATION_COLOR_HEX;
      switch (annotationMode) {
        case AnnotationMode.Highlighted:
          color = props.annotationColorHex || color;
          break;
        case AnnotationMode.All:
          color = (actorMap as any)[actor.track_id]?.color;
          break;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(actor.polygon.vertices[0].x * widthRatio, actor.polygon.vertices[0].y * heightRatio);
      // hacky draw, assumes already sorted
      actor.polygon.vertices.slice(1).forEach((vertex: Point) => {
        ctx.lineTo(vertex.x * widthRatio, vertex.y * heightRatio);
      });
      ctx.closePath();
      ctx.stroke();
    };

    const relativeTimestampMs = roundTimestamp(video.currentTime * 1000);
    const frameData = frameMap.getFrame(relativeTimestampMs);
    frameData?.actors.forEach((actor: Actor) => void drawBoundingBox(actor));

    setTimeout(draw, 33);
  }, [canvasRef, videoRef, actorMap, annotationMode, frameMap, props.annotationColorHex, props.cameraConfig]);

  useEffect(() => {
    if (readyToPlay) {
      videoRef?.current.addEventListener("play", draw);
      if (autoplay) {
        videoRef?.current.play().catch((e) => {
          console.error(e);
        });
      }
    }
  }, [readyToPlay, videoRef, draw, autoplay]);

  return (
    <div className={styles.playerWrapper}>
      {!readyToPlay && (
        <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 10 }}>
          <Spinner white />
        </Box>
      )}
      <div data-vjs-player>
        <video
          ref={videoRef}
          onCanPlay={props.onVideoCanPlay}
          muted
          playsInline
          className="video-js vjs-big-play-centered"
        />
        <canvas ref={canvasRef} className={styles.canvas} />
      </div>
    </div>
  );
};
