import { Text, View } from "@react-pdf/renderer";
import { IncidentTrend } from "features/dashboard";
import { DownArrow } from "./svgs/DownArrow";
import { UpArrow } from "./svgs/UpArrow";
import { styles } from "./styles";

interface TrendPercentageProps {
  trend: IncidentTrend;
}
export function TrendPercentage({ trend }: TrendPercentageProps) {
  if (trend.percentage === null) {
    return <Text style={[styles.tableRowText, { paddingRight: "3px" }]}>&#8211;</Text>;
  }
  const percent = trend.percentage as number;
  if (percent <= 0) {
    return (
      <View style={styles.percentage}>
        <DownArrow />
        <Text style={styles.tableRowText}>{Math.abs(percent)}%</Text>
      </View>
    );
  }
  if (percent > 0) {
    return (
      <View style={styles.percentage}>
        <UpArrow />
        <Text style={styles.tableRowText}>{Math.abs(percent)}%</Text>
      </View>
    );
  }
  return <></>;
}
