import { useMemo, useState } from "react";
import { OrgSite } from "features/executive-dashboard";
import { TrendCard } from "ui";
import { DateTime } from "luxon";
import { fillIncidentAggregateGroupsMergeIncidents } from "features/dashboard/components/trends/utils";
import { readableDaterange } from "shared/utilities/dateutil";
import { IncidentTrendSkeleton, IncidentType } from "features/dashboard";
import { SiteSummaryDataPanel } from "../dataPanels/SiteSummaryDataPanel";

interface TrendsBySiteProps {
  sites: OrgSite[];
  incidentTypes: IncidentType[];
  startDate: DateTime;
  endDate: DateTime;
  loading: boolean;
}
export function TrendsBySite({ sites, incidentTypes, startDate, endDate, loading }: TrendsBySiteProps) {
  const timezone = DateTime.now().zoneName;
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [dataPanelCurrentSite, setDataPanelCurrentSite] = useState<OrgSite>();

  const handleClick = (site: OrgSite) => {
    setDataPanelOpen(true);
    setDataPanelCurrentSite(site);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
    setDataPanelCurrentSite(undefined);
  };

  const trendCards = useMemo(() => {
    const trends = fillIncidentAggregateGroupsMergeIncidents(sites, incidentTypes, startDate, endDate, timezone, true);
    trends.sort((a, b) => b.countTotal - a.countTotal);
    const readable = readableDaterange(startDate, endDate, "to") || "";
    const dateText = `Total Incidents - ${readable}`;
    return trends.map((trend) => {
      const site = sites.find((s) => s.name === trend.name) as OrgSite;
      return (
        <TrendCard
          key={trend.name}
          trend={trend}
          title={trend.name}
          secondaryTitle={dateText}
          onClick={() => handleClick(site)}
        />
      );
    });
  }, [sites, incidentTypes, startDate, endDate, timezone]);

  return (
    <>
      {loading ? (
        <>
          <IncidentTrendSkeleton />
          <IncidentTrendSkeleton />
          <IncidentTrendSkeleton />
        </>
      ) : (
        <>
          {trendCards}
          {!!dataPanelCurrentSite && (
            <SiteSummaryDataPanel
              open={dataPanelOpen}
              startDate={startDate}
              endDate={endDate}
              site={dataPanelCurrentSite}
              onClose={handleDataPanelClose}
            />
          )}
        </>
      )}
    </>
  );
}
