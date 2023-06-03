import { Helmet } from "react-helmet-async";
import { Link } from "react-router-dom";
import classNames from "classnames";

interface Props {
  title: string;
  message: string;
}

export function GenericErrorPage({ title, message }: Props) {
  return (
    <>
      <Helmet>
        <title>{title}</title>
      </Helmet>
      <div className="h-full w-full grid items-center justify-center">
        <div style={{ maxWidth: 500 }} className="flex flex-col gap-4 p-4 text-center font-bold">
          <div className="text-3xl text-brand-gray-300">{title}</div>
          <div className="text-brand-gray-200">{message}</div>
          <div>
            <Link
              to="/"
              className={classNames(
                "inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium",
                "rounded-md shadow-sm text-white bg-brand-purple-500 hover:bg-brand-purple-700",
                "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-purple-500"
              )}
            >
              Go to Dashboard
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
