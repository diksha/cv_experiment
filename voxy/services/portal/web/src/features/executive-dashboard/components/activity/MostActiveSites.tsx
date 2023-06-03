import { TableCard, TableColumnDefinition } from "features/dashboard";
import { useMemo, useState } from "react";
import { DateTime } from "luxon";
import { readableDaterange } from "shared/utilities/dateutil";
import { OrgSite, SiteSession } from "features/executive-dashboard";
import { SiteSummaryDataPanel } from "../dataPanels";
import { TableSkeleton } from "ui";

interface RowData {
  siteName: string;
  value: number;
}
interface MostActiveSitesProps {
  siteSessions: SiteSession[];
  sites: OrgSite[];
  startDate: DateTime;
  endDate: DateTime;
  loading: boolean;
  showIcon?: boolean;
}
export function MostActiveSites({ siteSessions, sites, startDate, endDate, loading, showIcon }: MostActiveSitesProps) {
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [dataPanelCurrentSite, setDataPanelCurrentSite] = useState<OrgSite>();
  const dateText = readableDaterange(startDate, endDate, "to") || "";

  const columns: TableColumnDefinition<RowData, keyof RowData>[] = [
    {
      key: "siteName",
      header: "Site",
    },
    {
      key: "value",
      header: "Weekly Avg",
      textAlign: "right",
    },
  ];

  const rows: RowData[] = useMemo(() => {
    const results: RowData[] = [];

    siteSessions.forEach((item) => {
      results.push({
        siteName: item.site?.name || "",
        value: item.value,
      });
    });
    return results;
  }, [siteSessions]);

  const onClick = (row: RowData) => {
    const site = sites.find((s) => s.name === row.siteName) as OrgSite;
    setDataPanelOpen(true);
    setDataPanelCurrentSite(site);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
    setDataPanelCurrentSite(undefined);
  };

  if (loading) {
    return <TableSkeleton />;
  }

  return (
    <>
      <TableCard
        title="Most Active Sites"
        subtitle={`${dateText} - Weekly Avg Sessions per User`}
        data={rows}
        columns={columns}
        emptyMessage="No sessions during this time"
        onRowClick={onClick}
        uiKey="most-active-sites-card"
        icon="warehouse"
      />
      {!!dataPanelCurrentSite && (
        <SiteSummaryDataPanel
          open={dataPanelOpen}
          startDate={startDate as DateTime}
          endDate={endDate as DateTime}
          site={dataPanelCurrentSite}
          onClose={handleDataPanelClose}
        />
      )}
    </>
  );
}
