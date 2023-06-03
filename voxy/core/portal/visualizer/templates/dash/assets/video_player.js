/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable prefer-object-spread */
/* global videojs, frameAndActorMap, draw */
let callback;
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    load_video(videoLoaded) {
      if (videoLoaded) {
        return true;
      }
      const vjsPlayer = videojs("video", {
        controls: true,
        autoplay: true,
        playsinline: true,
        playbackRates: [0.5, 1, 1.5, 2, 3, 4, 5],
      });
      return true;
    },
    draw_labels(
      videoUrl,
      allLabels,
      labelChecklist,
      categoryChecklist,
      actorChecklist,
      attributesChecklist,
      cameraConfig
    ) {
      if (!videoUrl) return [[], [], []];
      const video = document.getElementById("video_html5_api");
      if (callback) {
        video.cancelVideoFrameCallback(callback);
      }
      const vjsPlayer = videojs("video", {
        controls: true,
        autoplay: true,
        playsinline: true,
      });
      vjsPlayer.src(videoUrl);
      const videoParent = document.getElementsByTagName("video")[0].parentNode;
      const canvas = document.getElementById("canvas_101");
      videoParent.insertBefore(canvas, videoParent.firstElementChild.nextElementSibling);
      const frameActorMapForLabel = {};
      allLabels.forEach(([labelKey, labelValue]) => {
        const [frameMapVal, actorMap] = frameAndActorMap(labelValue);
        frameActorMapForLabel[labelKey] = [frameMapVal, actorMap];
      });
      const attributeDom = document.getElementById("attribute_information");
      const doSomethingWithTheFrame = (now, metadata) => {
        const attributesMap = {};
        Object.entries(frameActorMapForLabel).forEach(([key, value]) => {
          if (!labelChecklist || labelChecklist.includes(key)) {
            const [frameMapVal, actorMap, categoriesList, attributesList] = value;
            const allActorsAttribute = draw(
              canvas,
              metadata,
              frameMapVal,
              actorMap,
              categoryChecklist,
              actorChecklist,
              attributesChecklist,
              cameraConfig
            );
            attributesMap[key] = allActorsAttribute;
          }
        });
        attributeDom.innerHTML = JSON.stringify(attributesMap, null, 2);
        callback = video.requestVideoFrameCallback(doSomethingWithTheFrame);
      };
      callback = video.requestVideoFrameCallback(doSomethingWithTheFrame);
      return [[]];
    },
  },
});
