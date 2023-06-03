export function isCurrentPath(pathname: string, to: string, extraMatches?: Array<string>): boolean {
  let active = pathname.startsWith(to);
  if (!active && extraMatches) {
    active = (extraMatches || []).some((path) => pathname.startsWith(path));
  }
  return active;
}
