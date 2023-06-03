import { Helmet } from "react-helmet-async";
import { Link } from "react-router-dom";
import { Logo } from "ui";
import classNames from "classnames";

export function Error404Page() {
  return (
    <>
      <Helmet>
        <title>Page not found - Voxel</title>
      </Helmet>
      <div className="bg-white min-h-full px-4 py-16 sm:px-6 sm:py-24 md:grid md:place-items-center lg:px-8">
        <div className="absolute top-0 left-0 w-24 p-4">
          <Link to="/">
            <Logo />
          </Link>
        </div>
        <div className="max-w-max mx-auto">
          <main className="sm:flex">
            <p className="text-4xl font-epilogue font-extrabold text-brand-purple-500 sm:text-5xl">404</p>
            <div className="sm:ml-6">
              <div className="sm:border-l sm:border-gray-200 sm:pl-6">
                <h1 className="text-4xl font-epilogue font-extrabold text-gray-900 tracking-tight sm:text-5xl">
                  Page not found
                </h1>
                <p className="mt-1 text-base text-gray-500">Please check the URL in the address bar and try again.</p>
              </div>
              <div className="mt-10 flex space-x-3 sm:border-l sm:border-transparent sm:pl-6">
                <Link
                  to="/"
                  className={classNames(
                    "inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium",
                    "rounded-md shadow-sm text-white bg-brand-purple-500 hover:bg-brand-purple-700",
                    "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-purple-500"
                  )}
                >
                  Go to home page
                </Link>
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
