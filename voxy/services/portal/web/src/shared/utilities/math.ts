export function average(array: number[]): number | undefined {
  const sum = array.reduce((a: number, b: number) => a + b, 0);
  return array.length > 0 ? sum / array.length : undefined;
}
