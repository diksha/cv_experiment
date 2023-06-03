import { IncidentAggregateGroup, IncidentAggregateGroupByDate } from "features/dashboard";
import { VoxelScore } from "shared/types";

export interface IncidentTrend {
  oneDayGroups: IncidentAggregateGroup[];
  mergedOneDayGroups: IncidentAggregateGroupByDate[];
  countTotal: number;
  percentage: number | null;
  name: string;
  incidentTypeKey: string;
}

export interface IncidentTrendWithScore extends IncidentTrend {
  score: VoxelScore | null;
}

export interface IncidentTrendsMapping {
  [incidentType: string]: IncidentTrend;
}

export interface HelperIncidentTrendsMapping {
  [incidentType: string]: {
    [date: string]: IncidentAggregateGroup[];
  };
}

interface Total {
  [id: string]: number;
}

export interface TotalsByDate {
  datetime: string;
  totals: Total;
}

interface IncidentTrendByCamera {
  oneDayGroups: IncidentAggregateGroup[];
  mergedOneDayGroups: IncidentAggregateGroupByDate[];
}

export interface IncidentTrendsByCameraMapping {
  [cameraId: string]: IncidentTrendByCamera;
}

export interface HelperIncidentTrendsByCameraMapping {
  [cameraId: string]: {
    [date: string]: IncidentAggregateGroup[];
  };
}
