import { FileDownload } from "@mui/icons-material";
import { Button, useTheme } from "@mui/material";
import { PDFDownloadLink } from "@react-pdf/renderer";
import { PdfReport } from "./PdfReport";
import { GetExecutiveDashboardData } from "__generated__/GetExecutiveDashboardData";
import { DateTime } from "luxon";

interface DownloadReportButtonProps {
  data: GetExecutiveDashboardData | undefined;
  startDate: DateTime;
  endDate: DateTime;
}

export function DownloadReportButton({ data, startDate, endDate }: DownloadReportButtonProps) {
  if (!data) {
    return <CustomButton loading={true} />;
  }
  return (
    <PDFDownloadLink
      document={<PdfReport data={data} startDate={startDate} endDate={endDate} />}
      fileName="VoxelReport.pdf"
    >
      {({ blob, url, loading, error }) => {
        return <CustomButton loading={loading} />;
      }}
    </PDFDownloadLink>
  );
}

interface CustomButtonProps {
  loading: boolean;
}

function CustomButton({ loading = true }: CustomButtonProps) {
  const theme = useTheme();

  return (
    <Button
      disabled={loading}
      variant="outlined"
      startIcon={<FileDownload />}
      sx={{
        border: `1px solid ${theme.palette.grey[300]}`,
        fontWeight: "400",
        height: "34px",
        borderRadius: "6px",
        lineHeight: "1.2",
      }}
      data-ui-key="dashboard-allsites-download-report"
    >
      Download Report
    </Button>
  );
}
