import React, { useState, useEffect } from "react";

import { BackgroundSpinner } from "ui";

declare const COMMIT_HASH: string;

export function DebugPage() {
  const [loading, setLoading] = useState(true);
  const [gitCommitHash, setGitCommitHash] = useState<string>("");
  const [hasError, setHasError] = useState<boolean>(false);

  useEffect(() => {
    const fetchCommitHash = async () => {
      try {
        const response = await fetch(`/api/commit_hash/`);
        const data = await response.json();
        setGitCommitHash(data);
      } catch (err) {
        console.error(err);
        setHasError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchCommitHash();
  }, [setGitCommitHash]);

  if (loading) {
    return <BackgroundSpinner />;
  }

  return (
    <div className="px-4 flex items-center justify-center h-screen w-full">
      <div className="m-auto flex flex-col items-center justify-center p-5 py-10 w-96">
        {!hasError ? (
          <p className="text-gray-600 text-center m-2">
            API Server Build Commit Hash: <span data-ui-key="api-commit-hash">{gitCommitHash}</span>
          </p>
        ) : null}
        <p className="text-gray-600 text-center m-2">
          Frontend Build Commit Hash: <span data-ui-key="frontend-commit-hash">{COMMIT_HASH}</span>
        </p>
      </div>
    </div>
  );
}
