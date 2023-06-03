import { useState, useEffect } from "react";
import NumberFormat from "react-number-format";
import classNames from "classnames";

interface StatisticBarProps {
  title: string;
  value: number;
  max?: number;
  barClassName?: string | null;
  barColor?: string | null;
  loading?: boolean;
  disabled?: boolean;
  uiKey?: string;
}

export function IncidentStatisticsBar(props: StatisticBarProps) {
  const [percentage, setPercentage] = useState<number>(100);

  useEffect(() => {
    if (props.value) {
      setPercentage((props.value / (props.max || props.value)) * 100);
    } else {
      setPercentage(100);
    }
  }, [props.value, props.max]);

  const barStyle = {
    width: `${percentage}%`,
    minWidth: 4,
    ...(props.value > 0 && props.barColor && { backgroundColor: props.barColor }),
  };

  return (
    <>
      {props.loading ? (
        <div className="bg-brand-gray-050 h-4 rounded-xl mb-4 animate-pulse"></div>
      ) : (
        <div
          data-ui-key={props.uiKey}
          className={classNames("rounded-lg p-2  border border-transparent ", {
            "hover:border-brand-gray-050 hover:shadow-lg": !props.disabled,
          })}
        >
          <div className={classNames("text-base", props.value > 0 ? "text-brand-gray-500" : "text-brand-gray-200")}>
            <span>{props.title}</span>
            <span className="font-bold ml-2">
              <NumberFormat value={props.value} thousandSeparator={true} displayType={"text"} />
            </span>
          </div>
          <div
            style={barStyle}
            className={classNames(
              props.value > 0 ? props.barClassName : "border border-brand-gray-050",
              "h-4 rounded-3xl mt-2"
            )}
          ></div>
        </div>
      )}
    </>
  );
}
