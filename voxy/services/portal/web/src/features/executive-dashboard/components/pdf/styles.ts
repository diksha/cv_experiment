import { StyleSheet } from "@react-pdf/renderer";

export const styles = StyleSheet.create({
  section: {
    marginBottom: "18px",
  },
  sectionTitle: {
    fontSize: "16px",
    fontWeight: "bold",
    marginBottom: "4px",
  },
  greyText: {
    fontSize: "12px",
    fontWeight: "thin",
    color: "#8b96a2",
  },
  tableHeaders: {
    display: "flex",
    flexDirection: "row",
    marginBottom: "4px",
  },
  tableHeaderBorder: {
    borderBottomColor: "black",
    borderBottomWidth: "1px",
  },
  tableRow: {
    display: "flex",
    flexDirection: "row",
    marginVertical: "6px",
  },
  tableRowText: {
    fontSize: "12px",
    fontWeight: "thin",
  },
  columnLeft: {
    flex: "2",
  },
  columnMiddle: {
    flex: "1",
  },
  columnRight: {
    flex: "1",
    textAlign: "right",
  },
  percentage: {
    display: "flex",
    flexDirection: "row-reverse",
    alignItems: "center",
  },
  flexRow: {
    display: "flex",
    flexDirection: "row",
  },
  scoreDash: {
    marginRight: "2px",
    position: "relative",
    bottom: "1px",
  },
});
