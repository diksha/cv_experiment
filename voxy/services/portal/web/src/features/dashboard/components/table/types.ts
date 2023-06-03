export type TableColumnDefinition<T, K extends keyof T> = {
  key: K;
  header: string;
  textAlign?: "left" | "center" | "right";
};

export type TableProps<T, K extends keyof T> = {
  data: Array<T>;
  columns: Array<TableColumnDefinition<T, K>>;
  uiKey: string;
  emptyMessage?: string;
  onRowClick?: (value: T) => void;
};
