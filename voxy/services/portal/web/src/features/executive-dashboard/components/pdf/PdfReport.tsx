import { ReactNode, useMemo } from "react";
import { Page, Text, View, Document } from "@react-pdf/renderer";
import { GetExecutiveDashboardData } from "__generated__/GetExecutiveDashboardData";
import { DateTime } from "luxon";
import { useTheme } from "@mui/material";
import { OrgSite, SiteSession, UserSession } from "features/executive-dashboard";
import { filterNullValues } from "shared/utilities/types";
import {
  fillIncidentAggregateGroupsMergeIncidents,
  fillIncidentAggregateGroupsMergeSitesIncidents,
  sortByTrendThenScore,
} from "features/dashboard/components/trends/utils";
import { VoxelScore } from "shared/types";
import { PDF_MAX_SITES_CHARS } from "features/executive-dashboard/constants";
import { Logo } from "./svgs/Logo";
import { styles } from "./styles";
import { TrendPercentage } from "./TrendPercentage";

interface PdfReportProps {
  data: GetExecutiveDashboardData;
  startDate: DateTime;
  endDate: DateTime;
}
export function PdfReport({ data, startDate, endDate }: PdfReportProps) {
  const theme = useTheme();
  const now = DateTime.now();
  const dateRangeText = `${startDate.toFormat("DDD")} to ${endDate.toFormat("DDD")}`;

  const sites = useMemo(() => {
    return filterNullValues<OrgSite>(data?.currentUser?.organization?.sites).filter((site) => site.isActive);
  }, [data]);
  const incidentTypes = useMemo(() => {
    return data?.currentUser?.organization?.incidentTypes || [];
  }, [data]);

  const orgRows = useMemo(() => {
    const mergeIncidentTypeKey = "merge";
    const mergedMap = fillIncidentAggregateGroupsMergeSitesIncidents(
      sites,
      incidentTypes,
      startDate,
      endDate,
      now.zoneName,
      true,
      mergeIncidentTypeKey
    );
    const score = data.currentUser?.organization?.overallScore?.value;
    return (
      <View style={styles.tableRow}>
        <View style={styles.columnLeft}>
          <Text style={styles.tableRowText}>{data.currentUser?.organization?.name}</Text>
        </View>
        <View style={[styles.columnMiddle, styles.flexRow]}>
          {typeof score === "number" ? (
            <Text style={styles.tableRowText}>{score}/100</Text>
          ) : (
            <Text style={[styles.tableRowText, styles.scoreDash]}>&#8211;</Text>
          )}
        </View>
        <View style={styles.columnRight}>
          <TrendPercentage trend={mergedMap[mergeIncidentTypeKey]} />
        </View>
      </View>
    );
  }, [data, startDate, endDate, incidentTypes, now.zoneName, sites]);

  const siteReportRows = useMemo(() => {
    const trends = fillIncidentAggregateGroupsMergeIncidents(
      sites,
      incidentTypes,
      startDate,
      endDate,
      now.zoneName,
      true
    );
    return trends.map((trend) => {
      const site = sites.find((s) => s.name === trend.name) as OrgSite;
      const score = site.overallScore?.value;
      return (
        <View key={site.id} style={styles.tableRow}>
          <View style={styles.columnLeft}>
            <Text style={styles.tableRowText}>{site.name}</Text>
          </View>
          <View style={[styles.columnMiddle, styles.flexRow]}>
            {typeof score === "number" ? (
              <Text style={styles.tableRowText}>{score}/100</Text>
            ) : (
              <Text style={[styles.tableRowText, styles.scoreDash]}>&#8211;</Text>
            )}
          </View>
          <View style={styles.columnRight}>
            <TrendPercentage trend={trend} />
          </View>
        </View>
      );
    });
  }, [startDate, endDate, incidentTypes, now.zoneName, sites]);

  const incidentReportRows = useMemo(() => {
    const mergedMap = fillIncidentAggregateGroupsMergeSitesIncidents(
      sites,
      incidentTypes,
      startDate,
      endDate,
      now.zoneName,
      true
    );
    const eventScoreData = filterNullValues<VoxelScore>(data?.currentUser?.organization?.eventScores);
    let sorted = sortByTrendThenScore(mergedMap, eventScoreData, incidentTypes);
    return sorted.map((item) => {
      return (
        <View key={item.incidentTypeKey} style={styles.tableRow}>
          <View style={styles.columnLeft}>
            <Text style={styles.tableRowText}>{item.name}</Text>
          </View>
          <View style={[styles.columnMiddle, styles.flexRow]}>
            {typeof item.score?.value === "number" ? (
              <Text style={styles.tableRowText}>{item.score.value}/100</Text>
            ) : (
              <Text style={[styles.tableRowText, styles.scoreDash]}>&#8211;</Text>
            )}
          </View>
          <View style={styles.columnRight}>
            <TrendPercentage trend={item} />
          </View>
        </View>
      );
    });
  }, [data, startDate, endDate, incidentTypes, now.zoneName, sites]);

  const activeSiteRows = useMemo(() => {
    const siteSessions = filterNullValues<SiteSession>(data?.currentUser?.organization?.sessionCount.sites).filter(
      (session) => session.site?.isActive
    );
    return siteSessions.map((session) => {
      return (
        <View key={session.site?.id} style={styles.tableRow}>
          <View style={styles.columnLeft}>
            <Text style={styles.tableRowText}>{session.site?.name}</Text>
          </View>
          <View style={styles.columnRight}>
            <View style={styles.percentage}>
              <Text style={styles.tableRowText}>{session.value}</Text>
            </View>
          </View>
        </View>
      );
    });
  }, [data]);

  const activeEmployeeRows = useMemo(() => {
    const userSessions = filterNullValues<UserSession>(data?.currentUser?.organization?.sessionCount.users);
    return userSessions.slice(0, 10).map((session) => {
      let sitesString =
        session.user?.sites
          ?.map((site) => {
            return site?.name;
          })
          .join(", ") || "";
      if (sitesString?.length > PDF_MAX_SITES_CHARS) {
        sitesString = sitesString?.slice(0, PDF_MAX_SITES_CHARS) + "...";
      }
      return (
        <View key={session.user?.id} style={styles.tableRow}>
          <View style={styles.columnLeft}>
            <Text style={styles.tableRowText}>{session.user?.fullName}</Text>
          </View>
          <View style={styles.columnMiddle}>
            <Text style={styles.tableRowText}>{sitesString}</Text>
          </View>
          <View style={styles.columnRight}>
            <View style={styles.percentage}>
              <Text style={styles.tableRowText}>{session.value}</Text>
            </View>
          </View>
        </View>
      );
    });
  }, [data]);

  return (
    <Document>
      <Page size="A4" style={{ padding: "36px" }}>
        <View style={styles.section}>
          <Logo />
          <Text style={{ fontSize: "10px", color: theme.palette.grey[500] }}>
            Report generated by {data.currentUser?.fullName} on {now.toFormat("DDD")} at {now.toFormat("t")}
          </Text>
        </View>

        <CustomSection
          title="Organization Summary"
          subtitle={dateRangeText}
          col1Header="Organization"
          col2Header="Voxel Score"
          col3Header="Total Incident Trend"
        >
          {orgRows}
        </CustomSection>

        <CustomSection
          title="Site Reports"
          subtitle={dateRangeText}
          col1Header="Site"
          col2Header="Voxel Score"
          col3Header="Total Incident Trend"
        >
          {siteReportRows}
        </CustomSection>

        <CustomSection
          title="Incident Reports"
          subtitle={dateRangeText}
          col1Header="Incident Type"
          col2Header="Voxel Score"
          col3Header="Total Incident Trend"
        >
          {incidentReportRows}
        </CustomSection>

        <CustomSection
          title="Most Active Sites"
          subtitle={dateRangeText}
          col1Header="Site"
          col3Header="Weekly Avg Sessions per User"
        >
          {activeSiteRows}
        </CustomSection>

        <CustomSection
          title="Most Active Employees"
          subtitle={`${dateRangeText} - Top 10 Employees`}
          col1Header="Employee"
          col2Header="Site"
          col3Header="Weekly Avg Sessions"
        >
          {activeEmployeeRows}
        </CustomSection>
      </Page>
    </Document>
  );
}

interface CustomSectionProps {
  title: string;
  subtitle: string;
  col1Header: string;
  col2Header?: string;
  col3Header: string;
  children: ReactNode;
}
function CustomSection({ title, subtitle, col1Header, col2Header, col3Header, children }: CustomSectionProps) {
  return (
    <>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{title}</Text>
        <Text style={styles.greyText}>{subtitle}</Text>
      </View>
      <View style={styles.section}>
        <View style={styles.tableHeaders}>
          <View style={styles.columnLeft}>
            <Text style={styles.greyText}>{col1Header}</Text>
          </View>
          {col2Header && (
            <View style={styles.columnMiddle}>
              <Text style={styles.greyText}>{col2Header}</Text>
            </View>
          )}
          <View style={styles.columnRight}>
            <Text style={styles.greyText}>{col3Header}</Text>
          </View>
        </View>
        <View style={styles.tableHeaderBorder}></View>
        {children}
      </View>
    </>
  );
}
