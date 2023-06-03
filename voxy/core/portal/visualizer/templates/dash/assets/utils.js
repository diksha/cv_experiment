/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable no-param-reassign */
/* global AVLTree, FrameMap */

function draw(
  canvas,
  metadata,
  frameMap,
  actorMap,
  categoryChecklist,
  actorChecklist,
  attributesChecklist,
  cameraConfig
) {
  const ctx = canvas.getContext("2d");
  canvas.width = metadata.width;
  canvas.height = metadata.height;
  const lineWidth = 5;
  const drawStaticPolygons = (polygons, fillColor) => {
    polygons.forEach((area) => {
      ctx.fillStyle = fillColor;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(area.polygon[0] * metadata.width, area.polygon[1] * metadata.height);
      // hacky draw, assumes already sorted
      area.polygon.forEach((vertex) => {
        ctx.lineTo(vertex[0] * metadata.width, vertex[1] * metadata.height);
      });
      ctx.closePath();
      ctx.fill();
    });
  };

  if (cameraConfig && cameraConfig["drivingAreas"]) {
    drawStaticPolygons(cameraConfig["drivingAreas"], "rgba(255,0,0,0.2)");
  }

  if (cameraConfig && cameraConfig["noPedestrianZones"]) {
    drawStaticPolygons(cameraConfig["noPedestrianZones"], "rgba(255,0,,0.2)");
  }
  if (cameraConfig && cameraConfig["doors"]) {
    drawStaticPolygons(cameraConfig["doors"], "rgba(255,0,0,0.2)");
  }
  if (cameraConfig && cameraConfig["motionDetectionZones"]) {
    drawStaticPolygons(cameraConfig["motionDetectionZones"], "rgba(255,0,0,0.2)");
  }

  function drawBoundingBox(actor) {
    if (actor === null || actor.polygon === null || actor.vertices === null) {
      return;
    }
    let color = "#E74C3C";
    if (actorMap[actor.track_id].color) {
      color = actorMap[actor.track_id].color;
    }
    ctx.font = "12px serif";
    ctx.fillText(actor.track_id, actor.polygon.vertices[0].x, actor.polygon.vertices[0].y);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(actor.polygon.vertices[0].x, actor.polygon.vertices[0].y);
    actor.polygon.vertices.slice(1).forEach((vertex) => {
      ctx.lineTo(vertex.x, vertex.y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  const relativeTimestampMs = Math.round(metadata.mediaTime * 1000);
  const frameData = frameMap.getFrame(relativeTimestampMs);
  const allActorsAttribute = {};
  if (frameData) {
    frameData.actors.forEach((actor) => {
      const ignoreActor =
        !actor ||
        (categoryChecklist && categoryChecklist.length && !categoryChecklist.includes(actor.category)) ||
        (actorChecklist && actorChecklist.length && !actorChecklist.includes(actor.track_id));
      if (ignoreActor) {
        return;
      }
      const actorAttribute = {};
      Object.entries(actor).forEach(([key, value]) => {
        if (!attributesChecklist || !attributesChecklist.length || attributesChecklist.includes(key)) {
          actorAttribute[key] = value;
        }
      });
      allActorsAttribute[actor.track_id] = actorAttribute;
      drawBoundingBox(actor);
    });
  }
  return allActorsAttribute;
}

const FrameMap = class {
  constructor() {
    this.timestampTree = new AVLTree();
    this.frameMap = {};
  }

  insertFrame(timestamp, frame) {
    this.timestampTree.insert(timestamp);
    this.frameMap[timestamp] = frame;
  }

  /**
   * Gets frame annotation data for the provided timestamp.
   */
  getFrame(timestamp) {
    // Traverse timestamp tree to find nearest matching timestamp
    const nearestTimestamp = this.timestampTree.findNearest(timestamp, "ceiling");
    // Return the matching frame data (if any)
    return this.frameMap[nearestTimestamp || -1];
  }
};

function randomColorString() {
  return `#${Math.floor(Math.random() * 16777215).toString(16)}`;
}

function generateColorStack() {
  const preferredColors = [
    "#E74C3C",
    "#8E44AD",
    "#5DADE2",
    "#48C9B0",
    "#F4D03F",
    "#E67E22",
    "#F5B7B1",
    "#D2B4DE",
    "#85C1E9",
    "#82E0AA",
    "#F7DC6F",
    "#EDBB99",
  ];

  const randomColors = [...Array(25)].map(() => randomColorString());
  return preferredColors.concat(randomColors);
}

function frameAndActorMap(annotations) {
  const frameAnnotations = annotations.frames;
  const frameMap = new FrameMap();
  const actorMap = {};
  const colors = generateColorStack();
  let timestampOffset = -1;
  if (frameAnnotations) {
    frameAnnotations.forEach((frame) => {
      // HACK: initialize offsets based on initial timestamp
      // Ideally this is handled in the controller/publisher
      if (timestampOffset === -1) {
        timestampOffset = frame.relative_timestamp_ms;
      }

      const relativeTimestampMs = Math.round(frame.relative_timestamp_ms - timestampOffset);

      frameMap.insertFrame(relativeTimestampMs, frame);
      // Generate an actor map with colors
      frame.actors.forEach((actor) => {
        if (!actorMap[actor.track_id]) {
          actorMap[actor.track_id] = {
            color: colors.shift(),
          };
        }
      });
    });
  }
  return [frameMap, actorMap];
}
