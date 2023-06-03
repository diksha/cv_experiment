import { Card } from "ui";
import { Link } from "react-router-dom";
import { IncidentStatisticsBar } from "features/incidents";

export interface IncidentStatisticsBarData {
  name: string;
  key: string;
  count: number;
  maxCount: number;
  barColor?: string | null;
  barClassName?: string | null;
  linkTo?: string;
}

interface IncidentStatisticsBarCardProps {
  title: string;
  data: IncidentStatisticsBarData[];
  loading: boolean;
  className?: string;
}

export function IncidentStatisticsBarCard({ title, data, loading, className }: IncidentStatisticsBarCardProps) {
  return (
    <Card className={className}>
      {loading ? (
        <LoadingSkeleton />
      ) : (
        <>
          <h4 className="text-brand-gray-500 pb-4">{title}</h4>
          {data.length > 0 ? (
            <div className="flex flex-col gap-1">
              {data?.map((item) => {
                const disabled = item.count === 0;
                const barEl = (
                  <IncidentStatisticsBar
                    key={item.key}
                    title={item.name}
                    value={item.count}
                    uiKey="button-filter-by-incident-type-statistic-bar"
                    max={item.maxCount || 0}
                    barColor={item.barColor}
                    disabled={disabled}
                  />
                );
                if (!disabled && item.linkTo) {
                  return (
                    <Link key={item.key} data-ui-key="button-filter-by-incident-type-statistic-link" to={item.linkTo}>
                      {barEl}
                    </Link>
                  );
                }
                return barEl;
              })}
              <div className="py-8 text-center text-brand-gray-200 hidden only:block">No incidents</div>
            </div>
          ) : (
            <div className="text-brand-gray-200">No data available</div>
          )}
        </>
      )}
    </Card>
  );
}

function LoadingSkeleton() {
  const loadingBars = Array.from(Array(6).keys());
  return (
    <>
      <div className="bg-brand-gray-050 h-6 w-56 rounded-xl mb-8 animate-pulse"></div>
      {loadingBars.map((key) => (
        <IncidentStatisticsBar loading={true} key={key} title={key.toString()} value={key} />
      ))}
    </>
  );
}
