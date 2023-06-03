import { Path, Svg } from "@react-pdf/renderer";

export function DownArrow() {
  return (
    <Svg viewBox="0 0 24 24" width="14px">
      <Path fill="#03bc1c" d="m20 12-1.41-1.41L13 16.17V4h-2v12.17l-5.58-5.59L4 12l8 8 8-8z"></Path>
    </Svg>
  );
}
