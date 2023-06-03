import { isEmpty } from "lodash";
import { useMemo } from "react";
import { TimeBucketWidth } from "shared/enums";
import { DowntimeCard } from "./DowntimeCard";
import { GetProductionLines_currentUser_site_productionLines } from "__generated__/GetProductionLines";
import { DateTime } from "luxon";
import { Box } from "@mui/material";

interface ProductionLine extends GetProductionLines_currentUser_site_productionLines {}

interface DowntimeProps {
  productionLines: ProductionLine[] | [];
  timezone: string;
  startDate?: DateTime;
  endDate?: DateTime;
}

export function Downtime({ productionLines, timezone, startDate, endDate }: DowntimeProps) {
  const alphaSorted: ProductionLine[] = useMemo(() => {
    if (!isEmpty(productionLines)) {
      return [...productionLines].sort((a, b) => a.name.localeCompare(b.name));
    }
    return [];
  }, [productionLines]);
  if (!isEmpty(productionLines)) {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {alphaSorted.map((productionLine) => (
          <DowntimeCard
            key={productionLine.id}
            productionLineId={productionLine.id}
            title={productionLine.name}
            data={productionLine.status1hGroups}
            timeBucketWidth={TimeBucketWidth.Hour}
            timezone={timezone}
          />
        ))}
      </Box>
    );
  }
  return null;
}
