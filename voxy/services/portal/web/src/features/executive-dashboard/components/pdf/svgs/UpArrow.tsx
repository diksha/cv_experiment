import { Path, Svg } from "@react-pdf/renderer";

export function UpArrow() {
  return (
    <Svg viewBox="0 0 24 24" width="14px">
      <Path fill="#bd0e08" d="m4 12 1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z"></Path>
    </Svg>
  );
}
