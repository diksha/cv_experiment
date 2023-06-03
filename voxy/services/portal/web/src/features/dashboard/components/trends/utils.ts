import {
  IncidentAggregateGroup,
  IncidentAggregateGroupByDate,
  IncidentType,
  Camera,
  PERIOD_MONTH,
  MIN_TREND_DAYS,
  PERIOD_WEEK,
  MIN_TREND_INCIDENT_COUNT,
} from "features/dashboard";
import { DateTime } from "luxon";
import { TimeBucketWidth } from "__generated__/globalTypes";
import {
  IncidentTrendsMapping,
  HelperIncidentTrendsMapping,
  IncidentTrend,
  IncidentTrendsByCameraMapping,
  HelperIncidentTrendsByCameraMapping,
  TotalsByDate,
  IncidentTrendWithScore,
} from "./types";
import { OrgSite } from "features/executive-dashboard";
import { isToday } from "shared/utilities/dateutil";
import { VoxelScore } from "shared/types";

type AcceptedBuckets = TimeBucketWidth.DAY | TimeBucketWidth.HOUR;
type IncidentTypeOmit = Omit<IncidentType, "__typename">;

const EMPTY = "empty";

const createEmptyAggregateGroup = (
  datetime: string,
  incidentKey: string,
  incidentName: string
): IncidentAggregateGroup => {
  return {
    id: "",
    dimensions: {
      datetime,
      incidentType: {
        id: "",
        key: incidentKey,
        name: incidentName,
        __typename: "OrganizationIncidentTypeType",
      },
      camera: {
        id: "",
        uuid: "",
        name: "",
        __typename: "CameraType",
      },
      __typename: "IncidentAggregateDimensions",
    },
    metrics: {
      count: 0,
      __typename: "IncidentAggregateMetrics",
    },
    __typename: "IncidentAggregateGroup",
  };
};

const createEmptyIncidentTrend = (name: string, key: string): IncidentTrend => {
  return {
    oneDayGroups: [],
    mergedOneDayGroups: [],
    countTotal: 0,
    percentage: null,
    name,
    incidentTypeKey: key,
  };
};

export const createHelperMap = (
  groups: IncidentAggregateGroup[],
  incidentTypes: IncidentTypeOmit[],
  isoDateOnly?: boolean,
  mergeIncidentTypeKey?: string
): [IncidentTrendsMapping, HelperIncidentTrendsMapping] => {
  const map: IncidentTrendsMapping = {};
  const helperMap: HelperIncidentTrendsMapping = {};

  incidentTypes.forEach((t) => {
    if (t?.key && t?.name) {
      map[mergeIncidentTypeKey || t.key] = createEmptyIncidentTrend(t.name, t.key);
      helperMap[mergeIncidentTypeKey || t.key] = {};
    }
  });

  groups.forEach((group) => {
    const date: string = isoDateOnly
      ? group.dimensions.datetime.match(/(20\d{2})-(\d{2})-(\d{2})/g)[0]
      : group.dimensions.datetime;
    const incidentType = mergeIncidentTypeKey || group.dimensions.incidentType.key;
    if (helperMap[incidentType]) {
      helperMap[incidentType][date] = helperMap[incidentType][date] || [];
      helperMap[incidentType][date].push(group);
    }
  });

  return [map, helperMap];
};

export const fillMap = <T extends IncidentTrendsByCameraMapping>(
  map: T,
  helperMap: HelperIncidentTrendsMapping | HelperIncidentTrendsByCameraMapping,
  groupBy: AcceptedBuckets,
  startDate: DateTime,
  endDate: DateTime,
  timezone: string,
  incidentTypeKey: string,
  incidentTypeName: string,
  isoDateOnly?: boolean
): T => {
  for (let key in helperMap) {
    let endTz: DateTime;
    let curDatetime: DateTime;
    if (groupBy === TimeBucketWidth.DAY) {
      endTz = endDate.setZone(timezone).set({ hour: 0, minute: 0, second: 0, millisecond: 0 });
      curDatetime = DateTime.fromISO(startDate.toISO())
        .setZone(timezone)
        .set({ hour: 0, minute: 0, second: 0, millisecond: 0 });
    } else if (groupBy === TimeBucketWidth.HOUR) {
      endTz = endDate.setZone(timezone).set({ minute: 0, second: 0, millisecond: 0 });
      curDatetime = DateTime.fromISO(startDate.toISO()).setZone(timezone).set({ minute: 0, second: 0, millisecond: 0 });
    } else {
      continue;
    }
    while (curDatetime <= endTz) {
      const iso = isoDateOnly ? curDatetime.toISODate() : curDatetime.toISO({ suppressMilliseconds: true });
      if (helperMap[key][iso]) {
        helperMap[key][iso].forEach((group) => {
          map[key].oneDayGroups.push(group);
          if ("countTotal" in map[key]) {
            const trend = map[key] as IncidentTrend;
            trend.countTotal += group.metrics.count;
          }
        });
      } else {
        const empty = createEmptyAggregateGroup(iso, incidentTypeKey, incidentTypeName);
        map[key].oneDayGroups.push(empty);
      }
      if (groupBy === TimeBucketWidth.DAY) {
        curDatetime = curDatetime.plus({ days: 1 });
      } else if (groupBy === TimeBucketWidth.HOUR) {
        curDatetime = curDatetime.plus({ hours: 1 });
      }
    }
  }
  for (let key in map) {
    map[key].mergedOneDayGroups = mergeSameDateIncidentAggregateGroups(map[key].oneDayGroups);
  }
  return map;
};

export const calculatePercentage = (map: IncidentTrendsMapping): IncidentTrendsMapping => {
  for (let key in map) {
    const trend = map[key];
    let curRollingAvg = 0;
    let prevRollingAvg = 0;
    let groupsLength = trend.mergedOneDayGroups.length;
    let period = 0;

    const lastGroupIsToday = isToday(
      DateTime.fromISO(trend.mergedOneDayGroups[groupsLength - 1].dimensions.datetime, { setZone: true })
    );
    if (lastGroupIsToday) {
      groupsLength--;
    }

    if (groupsLength >= PERIOD_MONTH * 2) {
      period = PERIOD_MONTH;
    } else if (groupsLength >= MIN_TREND_DAYS) {
      period = PERIOD_WEEK;
    }
    if (period && trend.countTotal >= MIN_TREND_INCIDENT_COUNT) {
      for (let i = groupsLength - 1; i >= 0; i--) {
        const ct = trend.mergedOneDayGroups[i].dateCount;
        if (i > groupsLength - 1 - period) {
          curRollingAvg += ct;
        }
        prevRollingAvg += ct;
      }
      curRollingAvg /= period;
      prevRollingAvg /= groupsLength;
      const change = Math.round(((curRollingAvg - prevRollingAvg) / prevRollingAvg) * 100);
      if (isFinite(change)) {
        trend.percentage = change;
      }
    }
  }
  return map;
};

export const fillIncidentAggregateGroups = (
  groups: IncidentAggregateGroup[],
  incidentTypes: IncidentTypeOmit[],
  groupBy: AcceptedBuckets = TimeBucketWidth.DAY,
  startDate: DateTime,
  endDate: DateTime,
  timezone: string,
  incidentTypeKey: string = EMPTY,
  incidentTypeName: string = EMPTY
): IncidentTrendsMapping => {
  let [map, helperMap] = createHelperMap(groups, incidentTypes);
  map = fillMap(map, helperMap, groupBy, startDate, endDate, timezone, incidentTypeKey, incidentTypeName);
  map = calculatePercentage(map);
  return map;
};

export const mergeSameDateIncidentAggregateGroups = (
  groups: IncidentAggregateGroup[]
): IncidentAggregateGroupByDate[] => {
  const result: IncidentAggregateGroupByDate[] = [];
  const sorted = groups.sort((a, b) => {
    return new Date(a.dimensions.datetime).getTime() - new Date(b.dimensions.datetime).getTime();
  });
  for (let i = 0; i < sorted.length; i++) {
    const clone = JSON.parse(JSON.stringify(sorted[i])) as IncidentAggregateGroupByDate;
    if (i === 0 || result[result.length - 1].dimensions.datetime !== sorted[i].dimensions.datetime) {
      clone.dateCount = clone.metrics.count;
      result.push(clone);
    } else {
      result[result.length - 1].dateCount += clone.metrics.count;
    }
  }
  return result;
};

export const fillIncidentAggregateGroupsByCamera = (
  groups: IncidentAggregateGroup[],
  cameras: Camera[],
  groupBy: AcceptedBuckets = TimeBucketWidth.DAY,
  startDate: DateTime,
  endDate: DateTime,
  timezone: string,
  incidentTypeKey: string = EMPTY,
  incidentTypeName: string = EMPTY
): TotalsByDate[] => {
  let [map, helperMap] = createHelperMapByCamera(groups, cameras, incidentTypeKey);
  map = fillMap(map, helperMap, groupBy, startDate, endDate, timezone, incidentTypeKey, incidentTypeName);
  const totals = groupTotalsByDate(map);
  return totals;
};

const createHelperMapByCamera = (
  groups: IncidentAggregateGroup[],
  cameras: Camera[],
  incidentTypeKey: string
): [IncidentTrendsByCameraMapping, HelperIncidentTrendsByCameraMapping] => {
  const map: IncidentTrendsByCameraMapping = {};
  const helperMap: HelperIncidentTrendsByCameraMapping = {};

  cameras.forEach((c) => {
    map[c.id] = {
      oneDayGroups: [],
      mergedOneDayGroups: [],
    };
    helperMap[c.id] = {};
  });

  groups.forEach((group) => {
    const cameraId = group.dimensions.camera.id;
    if (helperMap[cameraId] && group.dimensions.incidentType.key === incidentTypeKey) {
      helperMap[cameraId][group.dimensions.datetime] = helperMap[cameraId][group.dimensions.datetime] || [];
      helperMap[cameraId][group.dimensions.datetime].push(group);
    }
  });
  return [map, helperMap];
};

export const groupTotalsByDate = (map: IncidentTrendsByCameraMapping): TotalsByDate[] => {
  const result: TotalsByDate[] = [];
  for (let key in map) {
    for (let i = 0; i < map[key].mergedOneDayGroups.length; i++) {
      const current = map[key].mergedOneDayGroups[i];
      if (!result[i]) {
        result[i] = {
          datetime: current.dimensions.datetime,
          totals: {},
        };
      }
      result[i].totals[key] = current.dateCount;
    }
  }
  return result;
};

export const mergeIncidentTypes = (map: IncidentTrendsMapping, name: string): IncidentTrendsMapping => {
  const result: IncidentTrendsMapping = {
    merged: {
      oneDayGroups: [],
      mergedOneDayGroups: [],
      countTotal: 0,
      percentage: null,
      name: name,
      incidentTypeKey: "",
    },
  };
  for (let key in map) {
    const current = map[key];
    result.merged.countTotal += current.countTotal;
    if (!result.merged.mergedOneDayGroups.length) {
      result.merged.mergedOneDayGroups = current.mergedOneDayGroups;
    } else {
      for (let i = 0; i < result.merged.mergedOneDayGroups.length; i++) {
        result.merged.mergedOneDayGroups[i].dateCount += current.mergedOneDayGroups[i].dateCount;
      }
    }
  }
  return result;
};

export const mergeSitesAndIncidentTypes = (trendMaps: IncidentTrendsMapping[]): IncidentTrendsMapping => {
  let merged = trendMaps[0];
  for (let i = 1; i < trendMaps.length; i++) {
    for (let incidentType in trendMaps[i]) {
      for (let j = 0; j < trendMaps[i][incidentType].mergedOneDayGroups.length; j++) {
        const count = trendMaps[i][incidentType].mergedOneDayGroups[j].dateCount;
        merged[incidentType].countTotal += count;
        merged[incidentType].mergedOneDayGroups[j].dateCount += count;
      }
    }
  }
  return merged;
};

export const sortByTrendThenScore = (
  trendsMap: IncidentTrendsMapping,
  scores?: VoxelScore[],
  incidentTypes?: IncidentType[]
): IncidentTrendWithScore[] => {
  const [hasPercent, noPercent, hasScore, noScore]: IncidentTrendWithScore[][] = [...Array(4)].map((e) => []);
  for (let key in trendsMap) {
    let score = null;
    if (scores && incidentTypes) {
      const incidentType = incidentTypes.find((e) => e.name === trendsMap[key].name);
      // TODO(hq): compare key when score label becomes associated parent key
      score = scores.find((e) => e.label === incidentType?.name) || null;
    }
    const trend = {
      ...trendsMap[key],
      score,
    };
    if (trend.percentage !== null) {
      hasPercent.push(trend);
    } else {
      noPercent.push(trend);
    }
  }
  noPercent.forEach((trend) => {
    if (trend.score) {
      hasScore.push(trend);
    } else {
      noScore.push(trend);
    }
  });
  hasPercent.sort((a, b) => (b.percentage as number) - (a.percentage as number));
  hasScore.sort((a, b) => (b.score?.value as number) - (a.score?.value as number));
  return [...hasPercent, ...hasScore, ...noScore];
};

export function fillIncidentAggregateGroupsMergeSitesIncidents(
  sites: OrgSite[],
  incidentTypes: IncidentType[],
  startDate: DateTime,
  endDate: DateTime,
  timezone: string,
  isoDateOnly: boolean,
  mergeIncidentTypeKey: string = "",
  groupBy: AcceptedBuckets = TimeBucketWidth.DAY
): IncidentTrendsMapping {
  if (!sites.length) {
    const result = incidentTypes.reduce((map, str) => {
      map[mergeIncidentTypeKey || str.key] = createEmptyIncidentTrend(EMPTY, EMPTY);
      return map;
    }, {} as IncidentTrendsMapping);
    return result;
  }
  const trendMaps = sites.map((site) => {
    const groups = site?.incidentAnalytics?.incidentAggregateGroups || [];
    let [map, helperMap] = createHelperMap(groups, incidentTypes, isoDateOnly, mergeIncidentTypeKey);
    map = fillMap(map, helperMap, groupBy, startDate, endDate, timezone, mergeIncidentTypeKey, EMPTY, isoDateOnly);
    return map;
  });
  let mergedMap: IncidentTrendsMapping = mergeSitesAndIncidentTypes(trendMaps);
  mergedMap = calculatePercentage(mergedMap);
  return mergedMap;
}

export function fillIncidentAggregateGroupsMergeIncidents(
  sites: OrgSite[],
  incidentTypes: IncidentType[],
  startDate: DateTime,
  endDate: DateTime,
  timezone: string,
  isoDateOnly: boolean
): IncidentTrend[] {
  return sites.map((site) => {
    const groups = site?.incidentAnalytics?.incidentAggregateGroups || [];
    let [map, helperMap] = createHelperMap(groups, incidentTypes, isoDateOnly);
    map = fillMap(map, helperMap, TimeBucketWidth.DAY, startDate, endDate, timezone, EMPTY, EMPTY, isoDateOnly);
    map = mergeIncidentTypes(map, site.name);
    map = calculatePercentage(map);
    return map.merged;
  });
}
